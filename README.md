# Bonsai 本地助手

**一个双击就能用的本地 AI。** 没有账号，没有 API Key，没有命令行参数，对话不出这台电脑。

下载 → 双击 → 等模型下载完 → 开始聊。

> A local AI assistant that runs entirely on your own machine.
> One file to download, one switch to reach it from your phone.
> [English section below](#english)

---

## 它是什么

一个 Windows 桌面程序，把一个 270 亿参数的大模型跑在你自己电脑上，自带聊天界面。
第一次打开时它会自己下载模型（约 5.5 GB），之后就都是秒开。

- **完全本地**：对话、记录都在你的硬盘上，不经过任何服务器。
- **开箱即用**：双击图标就行。显存、线程、端口这些它是自己算的，不需要你决定。
- **手机也能用**：打开一个开关，它会给你一个临时网址和二维码，同一台机器以外的设备也能访问（走 Cloudflare 免费通道，不用注册）。
- **也能被程序调用**：自带 OpenAI 兼容接口，填上令牌就能接进你自己的工具。

## 怎么用

1. 到 [Releases](../../releases) 下载 `BonsaiLocal.exe`，放到任意文件夹。
2. 双击。第一次会显示下载进度（约 5.5 GB，取决于网速，一般十几分钟）。
3. 下载完就能聊了。

不需要装 Python、不需要装 CUDA、不需要改任何配置文件。

### 想用手机访问

1. 在电脑上打开 **设置 → 远程访问**，把开关打开。
2. 等几秒，会出现一个网址和二维码。
3. 手机扫码或输入网址，把**令牌**一并带上（设置页里可以复制）。

> ⚠️ 那个网址谁能打开，谁就能用你的模型。请只发给你信任的人，用完把开关关掉。
> 关掉后网址立即失效，下次打开会换一个新的。

### 接进别的程序

接口是 OpenAI 兼容的，地址 `http://127.0.0.1:<端口>/v1`，令牌在设置页里复制。

```bash
curl http://127.0.0.1:PORT/v1/chat/completions \
  -H "Authorization: Bearer sk-你的令牌" \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai","messages":[{"role":"user","content":"你好"}]}'
```

## 让它知道你不知道它不知道的事

27B 的模型对常识和写作很在行，但对**冷门事实**的记性很差 —— 具体到某个 CVE
编号是什么漏洞、某个 API 叫什么名字、某份内部文档里写了什么，它经常给出一个
听起来很专业、但其实编造的答案。

解决办法不是再找一个大模型，而是**把你自己的资料给它**：

**设置 → 我的资料 → 添加文件 / 添加文件夹**

支持 txt、md、PDF、以及各种代码和配置文件。加进去之后，它每次回答前会先在你
的资料里查一遍，把最相关的几段一起读，并在回答里标出依据来自哪份文件。
资料只存在本机，不会上传。

检索有两档，默认那档**零下载**：

| 档位 | 下载 | 擅长 |
|---|---|---|
| 关键词检索（默认） | 无 | CVE 编号、函数名、变量名这类精确词 |
| 更懂意思的检索 | 额外 610 MB | 换了说法也能查到、中英文互查 |

打开第二档会自己下载一个 0.6B 的向量模型，只下一次。

## 设置项只有四个

| 设置 | 说明 |
|---|---|
| 记忆容量 | 能记住多长的对话。三档，默认「标准」。 |
| 存储位置 | 模型和记录放哪。点「更改」会弹出系统的文件夹选择框。 |
| 远程访问 | 一个开关，附带网址和二维码。 |
| API 令牌 | 自动生成，可复制、可重新生成。 |

其它一切（显卡占用、线程数、上下文实现、推理参数）都是程序自己决定的，
因为绝大多数人没有办法对这些做出有意义的判断。

## 关于这个模型

底座是 [PrismML 的三值 Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
（Apache-2.0，每权重 1.75 bit）。它原本会对很多请求直接拒答；本程序在推理时叠加了
一个 8 MB 的秩-1 适配器，去掉了这个拒答行为。**也就是说，它会回答底座模型本来会
拒绝的请求。**

这个适配器只在「默认」以外的助手风格里生效。「默认」保持模型原本的样子 —— 这是
那个预设存在的意义。

这不是安全加固，恰恰相反。它是一个本地推理控制工具，请自行判断用它做什么。
生成内容的准确性与合法性由模型和使用者负责。

### 为什么不做「红队/逆向」之类的专业预设

我们试过。把五个安全方向的适配器（含一个我们自己转的、256/256 层全覆盖的红队
LoRA）在 31 个「懂行的人必须提到的点」上做过对照，最高的一个只命中 35.5%，而
`SeImpersonate`、`tcache`、`AmsiScanBuffer` 这类关键机制名**五个配置一次都没
出现过**；同时它们会编造 CVE 编号和不存在的 Metasploit 模块名，并把 PowerPC
汇编认成 MIPS。

结论很直接：**适配器能改的是"愿不愿意说"和"什么语气"，改不了"知不知道"。**
在专业领域，一个愿意说但不知道的模型比一个直接拒答的模型更危险 —— 它会把编造
的内容包装得像真的。所以这几个预设被撤掉了，取而代之的是上面的资料检索。

## 系统要求

- Windows 10/11 64 位
- 内存 16 GB 以上（强烈建议 32 GB）
- 想跑得快需要 NVIDIA 显卡；没有显卡也能跑，但会明显慢
- 磁盘预留 10 GB

首次运行会下载：

| 内容 | 大小 | 说明 |
|---|---|---|
| 模型 | 5.5 GB | 必需，只下一次 |
| 推理引擎（GPU） | 约 606 MB | 检测到 NVIDIA 显卡时下载（里面是 CUDA 运行时，所以偏大） |
| 推理引擎（CPU） | 约 8 MB | 没有独显时的兜底 |

## 常见问题

**下载中断了怎么办？** 直接重新双击。它会从断掉的地方继续，不会重头下。

**杀毒软件报警？** 单文件打包的程序常被误报。可以从源码自行构建：
`python build/make_release.py`。

**端口被占用？** 程序每次启动会重新挑一个可用端口，本地网址以窗口里显示的为准。

---

<a id="english"></a>
## English

**Bonsai Local** is a single-file Windows app that runs a 27B language model on your
own machine, with a chat interface and an OpenAI-compatible API.

1. Download `BonsaiLocal.exe` from [Releases](../../releases).
2. Double-click it. The first run downloads the model (~5.5 GB) and the matching
   inference engine for your hardware.
3. Chat. Nothing leaves your computer.

Flip **设置 → 远程访问** to get a temporary public URL (Cloudflare quick tunnel, no
account needed) plus a QR code. The API requires the token shown in settings; the
token is never embedded in the page when it is served over the tunnel, and all
settings endpoints are refused for non-local callers.

The bundled rank-1 adapter removes the base model's refusal behaviour. It is an
inference-control artifact, not a safety measure — see the note above.

## 来源与致谢

- 底座模型：[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)（Apache-2.0）
- 推理引擎：[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) fork（llama.cpp，MIT）
- 方向估计方法：Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024)
- 秩-1 适配器的构造方式参考了 [OrcaBonsai-27B-Uncensored](https://github.com/Continuum-AI-Corp/OrcaBonsai-27B-Uncensored)（Apache-2.0）

## 许可

本程序代码 MIT。模型与其适配器遵循各自的许可（Apache-2.0）。
