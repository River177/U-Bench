import os
import shutil
import subprocess

# ==================== 配置区 ====================

# 源模型：在哪个批次上训练的
SOURCE_BATCH = "bingren1-50"      # 这里就是你说的 A 批次
SOURCE_EXP_NAME = "default_exp"          # 源批次训练时用的 exp_name

# 要测试的 4 个批次
BATCHES = ["bingren1-50", "bingren3-17", "bingren4-13", "bingren5-11"]

# 模型和数据配置
MODEL_NAME = "MambaUnet"            # 跟训练时 --model 一致
DATASET_NAME = "bingren"          # 跟训练时 --dataset_name 一致
NUM_CLASSES = 7
IMG_SIZE = 256
GPU = "0"                         # 如果想用 CPU，可以改成 "cpu"

# 数据根目录
BASE_DIR_ROOT = r"data/bingren_processed_by_batch"

# Python 命令
PYTHON_CMD = "python"

# 跨批次测试专用的 exp_name（避免覆盖原训练结果）
def get_cross_exp_name(source_batch: str) -> str:
    # 例如 "from_bingren1-50"
    return f"from_{source_batch}"

# ==================== 工具函数 ====================

def get_exp_dir(batch_name: str, exp_name: str) -> str:
    """
    main_multi3d.py 的 init_dir 规则:
    exp_save_dir = ./output/{model}/{dataset_name}_{batch_name}/{exp_name}/
    """
    return os.path.join(
        "output",
        MODEL_NAME,
        f"{DATASET_NAME}_{batch_name}",
        exp_name,
    )

def ensure_source_checkpoint():
    """确认源批次自己的权重存在"""
    src_exp_dir = get_exp_dir(SOURCE_BATCH, SOURCE_EXP_NAME)
    src_ckpt = os.path.join(src_exp_dir, "checkpoint_best.pth")
    if not os.path.exists(src_ckpt):
        raise FileNotFoundError(
            f"找不到源批次 {SOURCE_BATCH} 的权重: {src_ckpt}\n"
            f"请先在该批次上完成训练，或检查 SOURCE_EXP_NAME 是否正确。"
        )
    print(f"[OK] 找到源批次权重: {src_ckpt}")
    return src_ckpt

def prepare_cross_test_exp_dir(target_batch: str, src_ckpt_path: str):
    """
    为目标批次创建一个“跨批次测试专用”的 exp 目录，并把源权重拷贝过去。
    目录形如:
      output/{MODEL}/bingren_{target_batch}/from_{SOURCE_BATCH}/checkpoint_best.pth

    不会动你原来训练用的 exp1 之类目录。
    """
    cross_exp_name = get_cross_exp_name(SOURCE_BATCH)
    tgt_exp_dir = get_exp_dir(target_batch, cross_exp_name)
    os.makedirs(tgt_exp_dir, exist_ok=True)

    tgt_ckpt = os.path.join(tgt_exp_dir, "checkpoint_best.pth")

    if os.path.exists(tgt_ckpt):
        print(f"[INFO] 目标批次 {target_batch} 的跨批次测试目录已存在权重: {tgt_ckpt}")
        print("       为避免覆盖，保留现有文件，不再拷贝。")
    else:
        print(f"[INFO] 拷贝源权重到跨批次测试目录:")
        print(f"      {src_ckpt_path}")
        print(f"   -> {tgt_ckpt}")
        shutil.copy2(src_ckpt_path, tgt_ckpt)

    return tgt_exp_dir

def run_cross_test_on_batch(target_batch: str):
    """
    使用“源批次 A 的权重”，在 target_batch 上跑 just_for_test。
    """
    base_dir = os.path.join(BASE_DIR_ROOT, target_batch)
    cross_exp_name = get_cross_exp_name(SOURCE_BATCH)

    cmd = [
        PYTHON_CMD,
        "main_multi3d.py",
        "--model", MODEL_NAME,
        "--base_dir", base_dir,
        "--dataset_name", DATASET_NAME,
        "--num_classes", str(NUM_CLASSES),
        "--img_size", str(IMG_SIZE),
        "--exp_name", cross_exp_name,   # 关键：使用跨批次专用 exp_name
        "--gpu", GPU,
        "--just_for_test", "True",
    ]

    print(f"\n[RUN] 使用 {SOURCE_BATCH} 的权重，在 {target_batch} 上测试")
    print("命令:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"在批次 {target_batch} 上测试失败，返回码 {result.returncode}")

def main():
    print("=== 跨批次泛化测试（固定源批次 A，用它的权重测 ABCDE） ===")
    print(f"源批次 (A): {SOURCE_BATCH}, 源 exp_name: {SOURCE_EXP_NAME}")
    print(f"目标批次列表 (ABCDE): {BATCHES}")

    # 1. 确认源权重存在
    src_ckpt_path = ensure_source_checkpoint()

    # 2. 为每个目标批次准备跨批次 exp 目录 + 拷贝源权重（不碰原训练权重）
    for batch in BATCHES:
        prepare_cross_test_exp_dir(batch, src_ckpt_path)

    # 3. 依次在每个批次上跑 just_for_test
    for batch in BATCHES:
        run_cross_test_on_batch(batch)

    print("\n=== 所有批次测试完成 ===")
    print("结果 CSV 会写在 ./result/ 下，文件名类似：")
    cross_exp_name = get_cross_exp_name(SOURCE_BATCH)
    for batch in BATCHES:
        # main_multi3d.py 里 CSV 命名是：result_{dataset_name}_test.csv
        # 目前不含批次名，但 exp_name 不同，日志和权重是分开的。
        print(f"  - 请查看: output/{MODEL_NAME}/{DATASET_NAME}_{batch}/{cross_exp_name}/ 里的 log 和 checkpoint")
    print("  - 以及: ./result/result_{dataset_name}_test.csv (会追加多行，包含不同批次的信息)")

if __name__ == "__main__":
    main()