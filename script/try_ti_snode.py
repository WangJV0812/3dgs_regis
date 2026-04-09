import taichi as ti

ti.init(arch=ti.cuda)

x = ti.field(ti.f32)

# === 修改开始 ===
# 1. 顶层：指针层。
# 维度被拆分了。总空间还是 (256, 256, 256, 1024)。
# 我们决定让底层块的大小为 (16, 16, 16, 16)。
# 那么顶层的尺寸就是：
# 256/16=16, 256/16=16, 256/16=16, 1024/16=64
# 顶层槽位数量：16*16*16*64 = 262,144 (非常小，几MB内存)
block = ti.root.pointer(ti.ijkl, (16, 16, 16, 64))

# 2. 底层：稠密层。
# 这一层真正存放数据。只有当顶层指针被激活时，这里才会分配 16^4 的连续内存块。
block.dense(ti.ijkl, (16, 16, 16, 16)).place(x)
# === 修改结束 ===

print('Successfully created the hierarchy!')

@ti.kernel
def fill():
    # 你的逻辑不变
    for i, j, k in ti.ndrange(100, 100, 100):
        for l in range(i):
            x[i, j, k, l] = i * 100 + j * 10 + k

@ti.kernel
def count_active_blocks():
    # 我们可以看看激活了多少个顶层指针块
    cnt = 0
    for I, J, K, L in block: # 遍历激活的 Block
        cnt += 1
    print("Active blocks:", cnt)

fill()
print('Successfully filled values')
count_active_blocks()