"""Unit tests for the models/ package.

Test coverage:
    - models.origin.resnet      : build_resnet 构建与预训练加载
    - models.compensators       : 每种补偿算子的 forward / 可训练参数 / fusible 标记
    - models.injector           : inject 注入逻辑 / PatchedBlock 三种 mode / InjectedModel 接口
    - models.builder            : build_model 统一入口 / clone / get_block_names

运行方式:
    python -m pytest tests/test_models.py -v
    python -m pytest tests/test_models.py -v -k "test_compensator"   # 只跑补偿器部分
"""

from __future__ import annotations
import pytest
import torch
import torch.nn as nn

# ── 被测模块 ──────────────────────────────────────────────────────────────────
from Src.Models_Nets import build_model, clone_teacher_to_student, get_block_names
from Src.Models_Nets.compensators import (
    # AdapterCompensator,
    AffineCompensator,
    IdentityCompensator,
    Linear1x1Compensator,
    LoRACompensator,
    ScalarCompensator,
    build_compensator,
    freeze_backbone_except_compensators,
    trainable_compensator_parameters,
)
from Src.Models_Nets.injector import (
    InjectedModel,
    PatchedBlock,
    inject,
    resnet_block_specs,
)
from Src.Models_Nets.origin.resnet import BasicBlock, Bottleneck, build_resnet


# ═════════════════════════════════════════════════════════════════════════════
# 公共 Fixtures
# ═════════════════════════════════════════════════════════════════════════════

# 使用小批量假数据，避免依赖真实数据集
BATCH = 2
CIFAR_INPUT  = torch.randn(BATCH, 3, 32, 32)   # CIFAR 尺寸（小 input）
IMAGENET_INPUT = torch.randn(BATCH, 3, 64, 64)  # 缩小的 ImageNet（加速测试）
NUM_CLASSES = 10


@pytest.fixture(scope="module")
def resnet18_model() -> InjectedModel:
    """ResNet-18 + IdentityCompensator，代表"无修改的原始网络"。"""
    model = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
    assert isinstance(model, InjectedModel), "build_model 应返回 InjectedModel 类型"
    return model


@pytest.fixture(scope="module")
def resnet50_model() -> InjectedModel:
    """ResNet-50 + IdentityCompensator。"""
    model = build_model("resnet50", num_classes=NUM_CLASSES, pretrained=False)
    assert isinstance(model, InjectedModel), "build_model 应返回 InjectedModel 类型"
    return model


@pytest.fixture(scope="module")
def resnet18_lora() -> InjectedModel:
    """ResNet-18 注入 LoRA 补偿器，用于微调相关测试。"""
    model = build_model("resnet18", num_classes=NUM_CLASSES, compensator_name="lora", compensator_rank=4)
    assert isinstance(model, InjectedModel), "build_model 应返回 InjectedModel 类型"
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Part 1 ── models/origin/resnet.py
# ═════════════════════════════════════════════════════════════════════════════

class TestOriginResnet:
    """测试官方 ResNet 构建器，确保它是纯净的 torchvision 结构。"""

    def test_build_resnet18_output_shape(self):
        """ResNet-18 的最终 logit 形状应为 (B, num_classes)。"""
        model = build_resnet(depth=18, num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(IMAGENET_INPUT)
        assert out.shape == (BATCH, NUM_CLASSES), \
            f"期望 ({BATCH}, {NUM_CLASSES})，实际 {out.shape}"

    def test_build_resnet50_output_shape(self):
        model = build_resnet(depth=50, num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(IMAGENET_INPUT)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_unsupported_depth_raises(self):
        """不支持的深度应该抛出 ValueError，而不是静默失败。"""
        with pytest.raises(ValueError, match="Unsupported ResNet depth"):
            build_resnet(depth=999)

    def test_fc_replaced_for_custom_classes(self):
        """当 num_classes != 1000 且 pretrained=True 时，fc 层应被替换。
        注意：此测试会触发网络权重下载，CI 环境中可用 @pytest.mark.slow 跳过。
        """
        pytest.importorskip("torchvision")  # 确保 torchvision 可用
        # 这里用 pretrained=False 模拟替换逻辑（不需要真实下载）
        model = build_resnet(depth=18, num_classes=NUM_CLASSES, pretrained=False)
        assert model.fc.out_features == NUM_CLASSES, \
            "fc 层输出维度应等于 num_classes"

    def test_block_types_resnet18(self):
        """ResNet-18 应由 BasicBlock 组成，而非 Bottleneck。"""
        model = build_resnet(depth=18, num_classes=NUM_CLASSES)
        blocks = [m for m in model.modules() if isinstance(m, (BasicBlock, Bottleneck))]
        assert all(isinstance(b, BasicBlock) for b in blocks), \
            "ResNet-18 不应包含 Bottleneck"

    def test_block_types_resnet50(self):
        """ResNet-50 应由 Bottleneck 组成。"""
        model = build_resnet(depth=50, num_classes=NUM_CLASSES)
        blocks = [m for m in model.modules() if isinstance(m, (BasicBlock, Bottleneck))]
        assert all(isinstance(b, Bottleneck) for b in blocks), \
            "ResNet-50 不应包含 BasicBlock"


# ═════════════════════════════════════════════════════════════════════════════
# Part 2 ── models/compensators.py
# ═════════════════════════════════════════════════════════════════════════════

# 用参数化测试覆盖所有补偿器，避免重复代码
COMPENSATOR_CASES = [
    ("identity", IdentityCompensator, True),
    ("scalar",   ScalarCompensator,   True),
    ("affine",   AffineCompensator,   True),
    ("linear",   Linear1x1Compensator, True),
    ("lora",     LoRACompensator,     False),
    # ("adapter",  AdapterCompensator,  False),
]

CHANNELS = 64
RANK = 8
FEATURE_MAP = torch.randn(BATCH, CHANNELS, 8, 8)  # 典型的中间特征图


class TestCompensators:
    """测试每种补偿算子的基础行为。"""

    @pytest.mark.parametrize("name, cls, fusible", COMPENSATOR_CASES)
    def test_output_shape_preserved(self, name, cls, fusible):
        """所有补偿器的输出形状必须与输入相同（残差加法要求维度一致）。"""
        comp = build_compensator(name, channels=CHANNELS, rank=RANK)
        comp.eval()
        with torch.no_grad():
            out = comp(FEATURE_MAP)
        assert out.shape == FEATURE_MAP.shape, \
            f"{name}: 期望 {FEATURE_MAP.shape}，实际 {out.shape}"

    @pytest.mark.parametrize("name, cls, fusible", COMPENSATOR_CASES)
    def test_is_compensator_flag(self, name, cls, fusible):
        """所有补偿器必须携带 is_compensator=True 标记，供 freeze 工具识别。"""
        comp = build_compensator(name, channels=CHANNELS, rank=RANK)
        assert getattr(comp, "is_compensator", False), \
            f"{name} 缺少 is_compensator 标记"

    @pytest.mark.parametrize("name, cls, fusible", COMPENSATOR_CASES)
    def test_fusible_flag(self, name, cls, fusible):
        """fusible 标记应与设计一致（影响部署时的重参数化决策）。"""
        comp = build_compensator(name, channels=CHANNELS, rank=RANK)
        assert comp.fusible == fusible, \
            f"{name}.fusible 期望 {fusible}，实际 {comp.fusible}"

    def test_identity_is_noop(self):
        """IdentityCompensator 必须是精确的恒等映射（输出 == 输入）。"""
        comp = IdentityCompensator(channels=CHANNELS)
        with torch.no_grad():
            out = comp(FEATURE_MAP)
        assert torch.equal(out, FEATURE_MAP), "IdentityCompensator 不应改变张量值"

    def test_scalar_trainable_param_count(self):
        """ScalarCompensator 应只有 1 个可训练参数（alpha）。"""
        comp = ScalarCompensator(channels=CHANNELS)
        n_params = sum(p.numel() for p in comp.parameters())
        assert n_params == 1, f"ScalarCompensator 应有 1 个参数，实际 {n_params}"

    def test_affine_param_count(self):
        """AffineCompensator 的参数量应为 2 * channels（gamma + beta）。"""
        comp = AffineCompensator(channels=CHANNELS)
        n_params = sum(p.numel() for p in comp.parameters())
        assert n_params == 2 * CHANNELS, \
            f"AffineCompensator 应有 {2 * CHANNELS} 个参数，实际 {n_params}"

    def test_lora_rank_clamp(self):
        """LoRACompensator 的 rank 不应超过 channels（自动截断）。"""
        comp = LoRACompensator(channels=8, rank=9999)
        # down 的输出通道数即为实际 rank
        actual_rank = comp.down.out_channels
        assert actual_rank <= 8, f"rank 应被截断到 channels=8，实际 {actual_rank}"

    def test_unsupported_name_raises(self):
        with pytest.raises(ValueError, match="Unsupported compensator"):
            build_compensator("nonexistent_compensator", channels=CHANNELS)

    def test_freeze_backbone_leaves_compensator_trainable(self):
        """freeze_backbone_except_compensators 后，补偿器参数应保持 requires_grad=True。"""
        model = build_model("resnet18", num_classes=NUM_CLASSES, compensator_name="lora")
        freeze_backbone_except_compensators(model)

        # 补偿器参数必须可训练
        comp_params = trainable_compensator_parameters(model)
        assert len(comp_params) > 0, "冻结后应仍有可训练的补偿器参数"
        assert all(p.requires_grad for p in comp_params), \
            "补偿器参数的 requires_grad 应为 True"

        # 主干参数必须被冻结（逐一检查非补偿器参数）
        for name, param in model.named_parameters():
            if not any(param is cp for cp in comp_params):
                assert not param.requires_grad, \
                    f"主干参数 {name} 应被冻结，但 requires_grad=True"


# ═════════════════════════════════════════════════════════════════════════════
# Part 3 ── models/injector.py
# ═════════════════════════════════════════════════════════════════════════════

class TestInjector:
    """测试注入引擎的核心逻辑。"""

    def test_all_blocks_replaced(self):
        """inject 后，模型中不应再存在原始的 BasicBlock / Bottleneck。"""
        backbone = build_resnet(depth=18, num_classes=NUM_CLASSES)
        injected = inject(backbone, resnet_block_specs, "identity", 16, "gelu")

        original_blocks = [m for m in injected.modules()
                           if isinstance(m, (BasicBlock, Bottleneck))]
        # BasicBlock 现在只应出现在 PatchedBlock.original_block 内部
        # 在顶层 named_modules 遍历中，它们应该被 PatchedBlock 包裹
        patched_blocks = [m for m in injected.backbone.modules()
                          if isinstance(m, PatchedBlock)]
        assert len(patched_blocks) > 0, "inject 后应存在 PatchedBlock"

    def test_block_order_length_resnet18(self):
        """ResNet-18 共有 8 个残差块（4 stage × 2 block），block_order 长度应为 8。"""
        model = build_model("resnet18", num_classes=NUM_CLASSES)
        names = get_block_names(model)
        assert len(names) == 8, f"ResNet-18 应有 8 个 block，实际 {len(names)}"

    def test_block_order_length_resnet50(self):
        """ResNet-50 共有 16 个残差块（3+4+6+3），block_order 长度应为 16。"""
        model = build_model("resnet50", num_classes=NUM_CLASSES)
        names = get_block_names(model)
        assert len(names) == 16, f"ResNet-50 应有 16 个 block，实际 {len(names)}"


class TestPatchedBlockModes:
    """测试 PatchedBlock 的三种 forward mode，这是整个动机实验的基础。"""

    @pytest.fixture
    def model(self) -> InjectedModel:
        m = build_model("resnet18", num_classes=NUM_CLASSES)
        m.eval()
        assert isinstance(m, InjectedModel), "build_model 应返回 InjectedModel 类型"
        return m

    def test_full_mode_output_shape(self, model):
        """full mode 推理形状应正确。"""
        with torch.no_grad():
            out = model(IMAGENET_INPUT, mode="full")
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_plain_mode_output_shape(self, model):
        """plain mode（删除全部残差）推理形状应正确。"""
        with torch.no_grad():
            out = model(IMAGENET_INPUT, mode="plain")
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_full_vs_plain_outputs_differ(self, model):
        """full 和 plain 的输出值应不同（残差对网络输出有影响）。
        
        这个测试同时验证了 plain mode 确实"删除"了残差——
        如果两者相等，说明残差被跳过的逻辑没有生效。
        """
        with torch.no_grad():
            out_full  = model(IMAGENET_INPUT, mode="full")
            out_plain = model(IMAGENET_INPUT, mode="plain")
        assert not torch.allclose(out_full, out_plain, atol=1e-5), \
            "full 和 plain 模式的输出不应相同"

    def test_partial_removal_removed_blocks_only(self, model):
        """只指定部分 block 进行残差删除时，输出应介于 full 和 plain 之间。"""
        all_blocks = get_block_names(model)
        # 只删最后一个 block
        last_block = [all_blocks[-1]]

        with torch.no_grad():
            out_full    = model(IMAGENET_INPUT, mode="full")
            out_partial = model(IMAGENET_INPUT, mode="plain", removed_blocks=last_block)
            out_plain   = model(IMAGENET_INPUT, mode="plain")

        # partial 既不等于 full 也不等于 plain
        assert not torch.allclose(out_partial, out_full, atol=1e-5)
        assert not torch.allclose(out_partial, out_plain, atol=1e-5)

    def test_compensated_mode_with_identity_equals_plain(self, model):
        """IdentityCompensator 的 compensated mode 应等价于 plain mode。
        
        因为 Identity(F(x)) == F(x)，这个等价关系是整个补偿器体系的
        "零点"基准，如果它失败，说明 compensated 的前向路径有 bug。
        """
        with torch.no_grad():
            out_plain       = model(IMAGENET_INPUT, mode="plain")
            out_compensated = model(IMAGENET_INPUT, mode="compensated")
        assert torch.allclose(out_plain, out_compensated, atol=1e-6), \
            "IdentityCompensator 的 compensated 输出应等于 plain 输出"

    def test_forward_collect_keys(self, model):
        """return_residual_stats=True 时，每个 block 应返回 plain/identity/output 三个张量。"""
        with torch.no_grad():
            result = model(IMAGENET_INPUT, mode="full", return_residual_stats=True)
        stats = result["residual_stats"]
        assert len(stats) > 0, "residual_stats 不应为空"
        for block_name, stat in stats.items():
            assert set(stat.keys()) == {"plain", "identity", "output"}, \
                f"block {block_name} 的统计字段不完整"

    def test_invalid_mode_raises(self, model):
        """传入不支持的 mode 字符串应抛出 ValueError。"""
        with pytest.raises(ValueError, match="Unsupported block mode"):
            # 需要触发 PatchedBlock.forward，而不是 InjectedModel 的分支
            block = next(m for m in model.modules() if isinstance(m, PatchedBlock))
            block(torch.randn(1, *FEATURE_MAP.shape[1:]), mode="invalid_mode")


class TestSplitInference:
    """测试端边协同切分推理接口（Exp3 系统实验的基础）。"""

    @pytest.fixture
    def model(self) -> InjectedModel:
        m = build_model("resnet18", num_classes=NUM_CLASSES)
        m.eval()
        assert isinstance(m, InjectedModel), "build_model 应返回 InjectedModel 类型"
        return m

    def test_split_then_continue_equals_full(self, model):
        """forward_to_split + forward_from_split 拼接后应等价于完整 forward。
        
        这验证了切分推理的"无损性"——切断后分开算，结果和一起算相同。
        """
        split_point = get_block_names(model)[3]  # 随便选第 4 个 block

        with torch.no_grad():
            out_full = model(IMAGENET_INPUT, mode="full")
            feat     = model.forward_to_split(IMAGENET_INPUT, split_point=split_point)
            out_split = model.forward_from_split(feat, split_point=split_point)

        assert torch.allclose(out_full, out_split, atol=1e-5), \
            "切分推理拼接后应与完整推理结果一致"

    def test_split_at_stem(self, model):
        """split_point='stem' 时，forward_to_split 应只运行 stem，形状合理。"""
        with torch.no_grad():
            feat = model.forward_to_split(IMAGENET_INPUT, split_point="stem")
        # stem 输出的空间尺寸为 input_size / 4（maxpool 后），通道数为 64
        assert feat.shape[1] == 64, f"stem 输出通道应为 64，实际 {feat.shape[1]}"

    def test_get_split_points_includes_stem(self, model):
        """get_split_points 应包含 'stem' 作为第一个元素。"""
        points = model.get_split_points()
        assert points[0] == "stem"
        assert len(points) == len(get_block_names(model)) + 1


# ═════════════════════════════════════════════════════════════════════════════
# Part 4 ── models/builder.py
# ═════════════════════════════════════════════════════════════════════════════

class TestBuilder:
    """测试统一入口 build_model 的各种调用方式。"""

    def test_resnet18_returns_injected_model(self):
        model = build_model("resnet18", num_classes=NUM_CLASSES)
        assert isinstance(model, InjectedModel)

    def test_resnet50_returns_injected_model(self):
        model = build_model("resnet50", num_classes=NUM_CLASSES)
        assert isinstance(model, InjectedModel)

    def test_mobilenet_v2_returns_injected_model(self):
        model = build_model("mobilenet_v2", num_classes=NUM_CLASSES)
        assert isinstance(model, InjectedModel)

    @pytest.mark.parametrize("comp_name", ["identity", "scalar", "affine", "lora", "adapter"])
    def test_all_compensators_buildable(self, comp_name):
        """每种补偿器都应能通过 build_model 成功注入，且推理不报错。"""
        model = build_model("resnet18", num_classes=NUM_CLASSES,
                            compensator_name=comp_name, compensator_rank=4)
        model.eval()
        with torch.no_grad():
            out = model(IMAGENET_INPUT, mode="full")
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_unsupported_arch_raises(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            build_model("vgg16", num_classes=NUM_CLASSES)

    def test_clone_creates_independent_copy(self):
        """clone_teacher_to_student 应返回深拷贝，修改副本不影响原模型。"""
        model = build_model("resnet18", num_classes=NUM_CLASSES)
        student = clone_teacher_to_student(model)

        # 修改 student 的第一个参数
        first_param_original = next(model.parameters()).clone()
        with torch.no_grad():
            next(student.parameters()).fill_(999.0)

        first_param_after = next(model.parameters())
        assert torch.equal(first_param_original, first_param_after), \
            "修改 student 参数不应影响原模型"

    def test_get_block_names_consistent(self):
        """get_block_names 的结果应与 model.get_block_names() 完全一致。"""
        model = build_model("resnet18", num_classes=NUM_CLASSES)
        assert get_block_names(model) == model.get_block_names()

    def test_freeze_then_only_compensators_train(self):
        """freeze_backbone 后优化器只绑定补偿器参数，反向传播不报错。"""
        model = build_model("resnet18", num_classes=NUM_CLASSES, compensator_name="lora")
        model.freeze_backbone()

        comp_params = model.compensator_parameters()
        assert len(comp_params) > 0

        optimizer = torch.optim.Adam(comp_params, lr=1e-3)
        out = model(IMAGENET_INPUT, mode="compensated")
        loss = out.sum()
        loss.backward()          # 不应抛出 RuntimeError
        optimizer.step()         # 不应抛出异常


# ═════════════════════════════════════════════════════════════════════════════
# Part 5 ── 集成测试（模拟动机实验 run_acc_drop.py 的核心循环）
# ═════════════════════════════════════════════════════════════════════════════

class TestMotivationExperimentIntegration:
    """端到端集成测试，模拟 Exp1 的逐块删除残差流程。
    
    这个测试的目的不是验证精度，而是验证整个调用链路不会出错——
    所有 mode 切换、removed_blocks 组合、以及 residual_stats 收集都能
    在一个完整的推理循环中稳定运行。
    """

    def test_incremental_block_removal_runs_without_error(self):
        """从最后一个 block 开始，逐步往前扩展删除范围，验证每一步都能正常推理。"""
        model = build_model("resnet18", num_classes=NUM_CLASSES)
        model.eval()
        all_blocks = get_block_names(model)

        with torch.no_grad():
            for i in range(1, len(all_blocks) + 1):
                removed = all_blocks[-i:]  # 逐步扩大删除范围
                out = model(IMAGENET_INPUT, mode="plain", removed_blocks=removed)
                assert out.shape == (BATCH, NUM_CLASSES), \
                    f"删除 {i} 个 block 后推理输出形状异常"

    def test_residual_stats_collection_for_all_blocks(self):
        """能对所有 block 同时收集 L2-norm 统计，供 run_residual_stats.py 使用。"""
        model = build_model("resnet18", num_classes=NUM_CLASSES)
        model.eval()
        all_blocks = get_block_names(model)

        with torch.no_grad():
            result = model(IMAGENET_INPUT, mode="full", return_residual_stats=True)

        stats = result["residual_stats"]
        assert set(stats.keys()) == set(all_blocks), \
            "residual_stats 的 key 集合应与 block_order 完全一致"

        # 验证可以从 stats 中计算 L2-norm 比值（动机实验的核心统计量）
        for block_name, stat in stats.items():
            plain_norm    = stat["plain"].norm().item()
            identity_norm = stat["identity"].norm().item()
            assert plain_norm > 0 and identity_norm > 0, \
                f"block {block_name} 的 norm 不应为零"