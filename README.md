**English Version**

---

## Project Overview

**TestDirchletFunctionMakePathosisHash** is a demonstration repository featuring two main components:

1. **Dirichlet Function Approximation**

   * The classic Dirichlet function

     $$
     D(x)=
     \begin{cases}
       1, & x \in \mathbb{Q},\\
       0, & x \notin \mathbb{Q}
     \end{cases}
     $$

     is not directly computable.
   * This project provides a C++ example (`TestDirchlet.cpp`) that approximates $D(x)$ by

     $$
     D_n(x) = \prod_{k=1}^n \bigl[\cos(k!\,\pi\,x)\bigr]^{2n},
     $$

     and prints its behavior on both rational and irrational test inputs to illustrate convergence toward 0 or 1.

2. **Pathosis Hash Algorithm**

   * “Pathosis” is a novel hash function design combining chaotic dynamics (a double‑pendulum system) with the Dirichlet approximation to enhance collision‑resistance and preimage‑resistance.
   * Full design, theory, and security analysis are documented in both Markdown and PDF:

     * `Pathosis - A Pathologically Secure Hash Function Based on Chaotic Dynamics and Dirichlet Approximation.md`
     * `Pathosis - A Pathologically Secure Hash Function Based on Chaotic Dynamics and Dirichlet Approximation.pdf`

---

## Repository Structure

```
/
├── LICENSE                                            # GPL-3.0 License
├── Pathosis - A Pathologically…Approximation.md       # Algorithm paper (Markdown)
├── Pathosis - A Pathologically…Approximation.pdf      # Algorithm paper (PDF)
├── TestDirchlet.cpp                                   # C++ example for Dirichlet approximation
├── TestDirchlet.sln                                   # Visual Studio solution
├── TestDirchlet.vcxproj*                              # VS project files
└── … other VS configuration files …
```

---

## Requirements & Build

* **OS**: Windows or Linux (with C++ toolchain)
* **Compiler**: C++17 or later (e.g., Visual Studio 2019+, g++ 9+)
* **Build**:

  * **Windows**: Open `TestDirchlet.sln` in Visual Studio and build.

---

## Usage

1. **Compile**

   * Windows: build via Visual Studio.
   * Command line:

     ```bash
     g++ TestDirchlet.cpp -o TestDirchlet -std=c++20 -O2
     ```

2. **Run**

   ```bash
   ./TestDirchlet
   ```

   Outputs approximations $D_n(x)$ for increasing $n$, demonstrating convergence.

3. **Read the Paper**

   * Open the Markdown or PDF to study Pathosis’s algorithmic design, mathematical rationale, chaotic mechanics, and security proofs.

---

## Highlights

* **Interdisciplinary**: Merges mathematical approximation with classical mechanics to form a robust hash.
* **Open Source**: Licensed under GPL‑3.0 for free use and modification.
* **Educational**: Clear example code plus in‑depth paper for further research or extension.

---

## Contribution

Feel free to open issues, submit pull requests, or leave comments for discussion and improvement.

---

---

**中文版本**

---

## 项目简介

**TestDirchletFunctionMakePathosisHash** 是一个演示性仓库，包含两个核心内容：

1. **Dirichlet 函数近似**

   * 经典的 Dirichlet 函数

     $$
     D(x)=
     \begin{cases}
       1, & x \in \mathbb{Q},\\
       0, & x \notin \mathbb{Q}
     \end{cases}
     $$

     无法直接计算。
   * 本项目提供了一个 C++ 示例（`TestDirchlet.cpp`），通过构造

     $$
     D_n(x) = \prod_{k=1}^n \bigl[\cos(k!\,\pi\,x)\bigr]^{2n}
     $$

     来数值近似 $D(x)$，并在有理数与无理数测试点上输出结果，展示其向 0 或 1 收敛的行为。

2. **Pathosis 哈希算法**

   * “Pathosis” 是一种新颖的哈希函数设计，结合了双摆系统的混沌动力学与 Dirichlet 近似，以提升抗碰撞性和抗预映像性。
   * 算法设计、理论基础和安全性分析已分别以 Markdown 和 PDF 形式完整记录：

     * `Pathosis - A Pathologically Secure Hash Function Based on Chaotic Dynamics and Dirichlet Approximation.md`
     * `Pathosis - A Pathologically Secure Hash Function Based on Chaotic Dynamics and Dirichlet Approximation.pdf`

---

## 目录结构

```
/
├── LICENSE                                            # GPL-3.0 开源协议
├── Pathosis - A Pathologically…Approximation.md       # 算法论文（Markdown）
├── Pathosis - A Pathologically…Approximation.pdf      # 算法论文（PDF）
├── TestDirchlet.cpp                                   # Dirichlet 近似示例代码
├── TestDirchlet.sln                                   # Visual Studio 解决方案
├── TestDirchlet.vcxproj*                              # VS 项目文件
└── … 其他 VS 配置文件 …
```

---

## 环境与依赖

* **操作系统**：Windows 或 Linux（需安装 C++ 编译工具链）
* **编译器**：C++17 或更高（如 Visual Studio 2019+、g++ 9+）
* **构建方式**：

  * Windows：使用 Visual Studio 打开 `TestDirchlet.sln` 并生成解决方案。

---

## 使用说明

1. **编译**

   * Windows：在 Visual Studio 中生成。
   * 命令行：

     ```bash
     g++ TestDirchlet.cpp -o TestDirchlet -std=c++20 -O2
     ```

2. **运行**

   ```bash
   ./TestDirchlet
   ```

   程序将打印不同 $n$ 下的 $D_n(x)$ 近似值，验证其收敛行为。

3. **阅读论文**

   * 打开 Markdown 或 PDF 文件，深入了解 Pathosis 算法设计、数学原理、混沌机制及安全性证明。

---

## 项目亮点

* **跨学科融合**：将数学近似与经典力学结合，构建高安全性的哈希函数。
* **开源自由**：GPL‑3.0 许可证，欢迎自由使用、修改与传播。
* **教学示例**：清晰的示例代码和详尽的论文，便于学习与二次开发。

---

## 贡献与交流

欢迎提交 Issue、Pull Request，或在仓库下留言，共同讨论改进与完善。
