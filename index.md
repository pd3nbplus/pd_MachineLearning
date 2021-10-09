<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-type" content="text/html;charset=UTF-8">
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"> </script>
<script type="text/x-mathjax-config">MathJax.Hub.Config({"tex2jax": {"inlineMath": [['$','$'], ['\\(','\\)']]}});</script>
<script type="text/x-mathjax-config">MathJax.Hub.Config({"HTML-CSS": {"availableFonts":["TeX"],"scale": 150}});</script>

<style>
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

body {
	font-family: "Segoe WPC", "Segoe UI", "SFUIText-Light", "HelveticaNeue-Light", sans-serif, "Droid Sans Fallback";
	font-size: 14px;
	padding: 0 12px;
	line-height: 22px;
	word-wrap: break-word;
}

#code-csp-warning {
	position: fixed;
	top: 0;
	right: 0;
	color: white;
	margin: 16px;
	text-align: center;
	font-size: 12px;
	font-family: sans-serif;
	background-color:#444444;
	cursor: pointer;
	padding: 6px;
	box-shadow: 1px 1px 1px rgba(0,0,0,.25);
}

#code-csp-warning:hover {
	text-decoration: none;
	background-color:#007acc;
	box-shadow: 2px 2px 2px rgba(0,0,0,.25);
}


body.scrollBeyondLastLine {
	margin-bottom: calc(100vh - 22px);
}

body.showEditorSelection .code-line {
	position: relative;
}

body.showEditorSelection .code-active-line:before,
body.showEditorSelection .code-line:hover:before {
	content: "";
	display: block;
	position: absolute;
	top: 0;
	left: -12px;
	height: 100%;
}

body.showEditorSelection li.code-active-line:before,
body.showEditorSelection li.code-line:hover:before {
	left: -30px;
}

.vscode-light.showEditorSelection .code-active-line:before {
	border-left: 3px solid rgba(0, 0, 0, 0.15);
}

.vscode-light.showEditorSelection .code-line:hover:before {
	border-left: 3px solid rgba(0, 0, 0, 0.40);
}

.vscode-dark.showEditorSelection .code-active-line:before {
	border-left: 3px solid rgba(255, 255, 255, 0.4);
}

.vscode-dark.showEditorSelection .code-line:hover:before {
	border-left: 3px solid rgba(255, 255, 255, 0.60);
}

.vscode-high-contrast.showEditorSelection .code-active-line:before {
	border-left: 3px solid rgba(255, 160, 0, 0.7);
}

.vscode-high-contrast.showEditorSelection .code-line:hover:before {
	border-left: 3px solid rgba(255, 160, 0, 1);
}

img {
	max-width: 100%;
	max-height: 100%;
}

a {
	color: #4080D0;
	text-decoration: none;
}

a:focus,
input:focus,
select:focus,
textarea:focus {
	outline: 1px solid -webkit-focus-ring-color;
	outline-offset: -1px;
}

hr {
	border: 0;
	height: 2px;
	border-bottom: 2px solid;
}

h1 {
	padding-bottom: 0.3em;
	line-height: 1.2;
	border-bottom-width: 1px;
	border-bottom-style: solid;
}

h1, h2, h3 {
	font-weight: normal;
}

h1 code,
h2 code,
h3 code,
h4 code,
h5 code,
h6 code {
	font-size: inherit;
	line-height: auto;
}

a:hover {
	color: #4080D0;
	text-decoration: underline;
}

table {
	border-collapse: collapse;
}

table > thead > tr > th {
	text-align: left;
	border-bottom: 1px solid;
}

table > thead > tr > th,
table > thead > tr > td,
table > tbody > tr > th,
table > tbody > tr > td {
	padding: 5px 10px;
}

table > tbody > tr + tr > td {
	border-top: 1px solid;
}

blockquote {
	margin: 0 7px 0 5px;
	padding: 0 16px 0 10px;
	border-left: 5px solid;
}

code {
	font-family: Menlo, Monaco, Consolas, "Droid Sans Mono", "Courier New", monospace, "Droid Sans Fallback";
	font-size: 14px;
	line-height: 19px;
}

body.wordWrap pre {
	white-space: pre-wrap;
}

.mac code {
	font-size: 12px;
	line-height: 18px;
}

pre:not(.hljs),
pre.hljs code > div {
	padding: 16px;
	border-radius: 3px;
	overflow: auto;
}

/** Theming */

.vscode-light,
.vscode-light pre code {
	color: rgb(30, 30, 30);
}

.vscode-dark,
.vscode-dark pre code {
	color: #DDD;
}

.vscode-high-contrast,
.vscode-high-contrast pre code {
	color: white;
}

.vscode-light code {
	color: #A31515;
}

.vscode-dark code {
	color: #D7BA7D;
}

.vscode-light pre:not(.hljs),
.vscode-light code > div {
	background-color: rgba(220, 220, 220, 0.4);
}

.vscode-dark pre:not(.hljs),
.vscode-dark code > div {
	background-color: rgba(10, 10, 10, 0.4);
}

.vscode-high-contrast pre:not(.hljs),
.vscode-high-contrast code > div {
	background-color: rgb(0, 0, 0);
}

.vscode-high-contrast h1 {
	border-color: rgb(0, 0, 0);
}

.vscode-light table > thead > tr > th {
	border-color: rgba(0, 0, 0, 0.69);
}

.vscode-dark table > thead > tr > th {
	border-color: rgba(255, 255, 255, 0.69);
}

.vscode-light h1,
.vscode-light hr,
.vscode-light table > tbody > tr + tr > td {
	border-color: rgba(0, 0, 0, 0.18);
}

.vscode-dark h1,
.vscode-dark hr,
.vscode-dark table > tbody > tr + tr > td {
	border-color: rgba(255, 255, 255, 0.18);
}

.vscode-light blockquote,
.vscode-dark blockquote {
	background: rgba(127, 127, 127, 0.1);
	border-color: rgba(0, 122, 204, 0.5);
}

.vscode-high-contrast blockquote {
	background: transparent;
	border-color: #fff;
}
</style>

<style>
/* Tomorrow Theme */
/* http://jmblog.github.com/color-themes-for-google-code-highlightjs */
/* Original theme - https://github.com/chriskempson/tomorrow-theme */

/* Tomorrow Comment */
.hljs-comment,
.hljs-quote {
	color: #8e908c;
}

/* Tomorrow Red */
.hljs-variable,
.hljs-template-variable,
.hljs-tag,
.hljs-name,
.hljs-selector-id,
.hljs-selector-class,
.hljs-regexp,
.hljs-deletion {
	color: #c82829;
}

/* Tomorrow Orange */
.hljs-number,
.hljs-built_in,
.hljs-builtin-name,
.hljs-literal,
.hljs-type,
.hljs-params,
.hljs-meta,
.hljs-link {
	color: #f5871f;
}

/* Tomorrow Yellow */
.hljs-attribute {
	color: #eab700;
}

/* Tomorrow Green */
.hljs-string,
.hljs-symbol,
.hljs-bullet,
.hljs-addition {
	color: #718c00;
}

/* Tomorrow Blue */
.hljs-title,
.hljs-section {
	color: #4271ae;
}

/* Tomorrow Purple */
.hljs-keyword,
.hljs-selector-tag {
	color: #8959a8;
}

.hljs {
	display: block;
	overflow-x: auto;
	color: #4d4d4c;
	padding: 0.5em;
}

.hljs-emphasis {
	font-style: italic;
}

.hljs-strong {
	font-weight: bold;
}
</style>

<style>
/*
 * Markdown PDF CSS
 */

pre {
	background-color: #f8f8f8;
	border: 1px solid #cccccc;
	border-radius: 3px;
	overflow-x: auto;
	white-space: pre-wrap;
	overflow-wrap: break-word;
}

pre:not(.hljs) {
	padding: 23px;
	line-height: 19px;
}

blockquote {
	background: rgba(127, 127, 127, 0.1);
	border-color: rgba(0, 122, 204, 0.5);
}

.emoji {
	height: 1.4em;
}

/* for inline code */
:not(pre):not(.hljs) > code {
	color: #C9AE75; /* Change the old color so it seems less like an error */
	font-size: inherit;
}

</style>

</head>
<body>
<h1 id="mechaine-learning--%E6%BD%98%E7%99%BB%E5%90%8C%E5%AD%A6%E7%9A%84%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0">Mechaine Learning--潘登同学的机器学习笔记</h1>
<h1 id="%E7%9B%AE%E5%BD%95">目录</h1>
<ul>
<li><a href="#index1">有监督机器学习</a>
<ul>
<li>多元线性回归</li>
<li></li>
</ul>
</li>
</ul>
<h1 id="%E6%9C%89%E7%9B%91%E7%9D%A3%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0index1">有监督机器学习{#index1}</h1>
<h2 id="%E5%A4%9A%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92mlr">多元线性回归(MLR)</h2>
<ul>
<li>
<p><code>总目标</code>：预测</p>
</li>
<li>
<p><code>模型：</code>
$$
y = \beta_0 + \beta_1x_1 + \cdots + \beta_kx_k
$$</p>
</li>
<li>
<p><code>优化目标：</code>预测的越精确越好, 预测误差(残差)越小越好</p>
</li>
</ul>
<blockquote>
<p>Actual value:真实的y值(样本中自带的)</p>
<p>Predicted value:预测的y值(通过多元线性回归得出的y,通常记作 $\hat{y}$</p>
<p>error:预测值与真实值的误差(error = $\hat{y} - y$)</p>
<p>loss:表示我们最优化的目标,我们希望残差越小越好</p>
</blockquote>
<p>$$
Loss = \sum_{n=1}^{m} error^{2} = \sum_{n=1}^{m} (\hat{y} - y)^{2}
$$</p>
<ul>
<li><code>求解目标：</code>
$$
\min_{\beta_0,\beta_0,\ldots,\beta_k} Loss = \min_{\beta_0,\beta_0,\ldots,\beta_k} \sum_{n=1}^{m} error^{2} = \min_{\beta_0,\beta_0,\ldots,\beta_k} \sum_{n=1}^{m} (\hat{y} - y)^{2}
$$
上述目标称为MSE(Mean Square Error),我们习惯用$\theta$来代替这一串${\beta_0,\beta_0,\ldots,\beta_k}$</li>
</ul>
<h3 id="%E7%94%B1%E6%9E%81%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1mle-maximum-likelihood-estimation%E6%8E%A8%E5%AF%BCmse">由极大似然估计(MLE, Maximum likelihood estimation)推导MSE</h3>
<ul>
<li>理解为什么$\theta$不是唯一的</li>
</ul>
<blockquote>
<p>因为数据都是从总体中抽样出来的,只要我们的样本不同,${\theta}$就不会相同,我们将特定样本计算出的${\theta}$称为$\hat{\theta}$,所以我们只能得到$\hat{\theta}$的无偏估计,而不能得到${\theta}$的无偏估计(除非我们的样本就是总体);</p>
</blockquote>
<p>问题1: 那么我们凭什么能用样本计算出的$\hat{\theta}$来代替${\theta}$</p>
<ul>
<li>解决上面的问题, 采用中心极限定理
<img src="file:///c:/Users/潘登/Documents/GitHub/pd_MachineLearning/img/中心极限定理.jpg" alt="pd的中心极限理解"></li>
</ul>
<p>问题2: 中心极限定理又跟$\hat{\theta}$可以当作${\theta}$有什么关系呢？</p>
<ul>
<li>构造似然函数,最大化总似然</li>
</ul>
<p>假若说我们已经对样本进行了模型求解,那么我们得到了某个具体的$\hat{\theta_1}$</p>
<p>那么这个$\hat{\theta_1}$到底有多像${\theta}$呢？</p>
<blockquote>
<p>给定一个概率分布 D，已知其概率密度函数（连续分布）或概率质量函数（离散分布）为 f_D，以及一个分布参数θ，我们可以从这个分布中抽出一个具有 m 个值的采样 $x_1,x_2,\ldots,x_m$,利用f_D计算似然函数：
$$ L(\theta|x_1,x_2,\ldots,x_m) = f_{\theta}(x_1,x_2,\ldots,x_m)$$</p>
<p>若 D 是离散分布，$f_{\theta}$ 即是在参数为θ时观测到这一采样的概率。若其是连续分布，$f_{\theta}$ 则为 $x_1,x_2,\ldots,x_m$ 联合分布的概率密度函数在观测值处的取值。</p>
<p>一旦我们获得了$x_1,x_2,\ldots,x_m$,那么我们就能获得一个关于${\theta}$的估计(也就是$\hat{\theta}$)</p>
<p>最大似然估计会寻找关于${\theta}$的最可能的值（即，在所有可能 的${\theta}$取值中，寻找一个值使这个采样的“可能性”最大化）。从数学上来说，我们可以在${\theta}$的所有可能取值中寻找一个值使得<code>似然函数取到最大值</code>。这个使可能性最大的$\hat{\theta}$值即称为${\theta}$的最大似然估计。由定义，最大似然估计是关于样本的函数。</p>
</blockquote>
<p>问题3：那这个概率密度函数是什么呢？</p>
<blockquote>
<p>由中心极限定理,这个概率密度函数是正态分布,其概率密度函数为
$$ f(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}} $$</p>
</blockquote>
<p>但是我们需要的是估计${\theta}$, 我们转变一下思路,只要在给定样本$X_{m\times n}$,计算出$\hat{\theta}$之后, 其实$\hat{y}$就知道了, 那么误差项 $\varepsilon$就能知道了, 且$\varepsilon$的大小其实跟$\hat{\theta}$准不准确是有极大关联的(很好理解，$\hat{\theta}$越准确,$\varepsilon$越小, 这个可以证明);</p>
<p>所以我们可以用$\varepsilon$来做最大似然估计
$$
f(\varepsilon_i|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(\varepsilon_i-\mu)^2}{2\sigma^2}}
$$</p>
<p>而$\varepsilon$的均值为0, 这个其实是我们的一项假设, 但是它其实也不是假设, 当我们完成我们的优化目标时, 这个均值假设其实就变成一个可以推导出的结论;所以上式改写成：
$$
f(\varepsilon_i|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(\varepsilon_i-0)^2}{2\sigma^2}}
$$</p>
<p>那么我们的似然函数就可以表示为：
$$
L_\theta(\varepsilon_1, \varepsilon_2, \ldots, \varepsilon_m)=f(\varepsilon_1, \varepsilon_2, \ldots, \varepsilon_m|\mu,\sigma^2)
$$</p>
<p>又因为残差$\varepsilon$服从正态分布, 自然暗含了相互独立的假设,可以把上面的式子写成连乘的形式：
$$
f(\varepsilon_1, \varepsilon_2, \ldots, \varepsilon_m|\mu,\sigma^2)=f(\varepsilon_1|\mu,\sigma^2)<em>f(\varepsilon_2|\mu,\sigma^2)</em>\cdots*f(\varepsilon_m|\mu,\sigma^2)
$$
进而有：
$$
L_\theta(\varepsilon_1, \varepsilon_2, \ldots, \varepsilon_m) = \prod_{i=1}^{m} f(\varepsilon_i|\mu,\sigma^2) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(\varepsilon_i-0)^2}{2\sigma^2}}
$$</p>
<p>把$x_i$和$y_i$带入上式:
$$
\varepsilon_i = \hat{y} - y = \theta x_i - y_i\
L_\theta(\varepsilon_1, \varepsilon_2, \ldots, \varepsilon_m) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(\theta ^{T}x_i - y_i)^2}{2\sigma^2}}
$$
<code>最大化总似然:</code>
$$
\argmax_{\theta} L_\theta(\varepsilon_1, \varepsilon_2, \ldots, \varepsilon_m) = \argmax_{\theta} \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(\theta ^{T}x_i - y_i)^2}{2\sigma^2}}
$$</p>
<ul>
<li>对数似然函数</li>
</ul>
<p>对上面总似然函数取对数,得到的函数记为为${\Bbb{L}}$：
$$
{\Bbb{L}} = \argmax_{\theta} log_e(\prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(\theta ^{T}x_i - y_i)^2}{2\sigma^2}})=\argmax_{\theta} \sum_{i=1}^{m}\log_e(\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(\theta ^{T}x_i - y_i)^2}{2\sigma^2}})
$$</p>
<ul>
<li>继续往下推(这里省略了,因为只是体力活而已)
$$
{\Bbb{L}} = \argmax_{\theta} m\bullet \ln \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{1}{\sigma^2}\bullet\frac{1}{2}\bullet \sum_{i=1}^{m}(\theta ^{T}x_i - y_i)^2
$$
那么前面一项是常数, 最大化${\Bbb{L}}$就是最小化$\sum_{i=1}^{m}(\theta ^{T}x_i - y_i)^2$</li>
</ul>
<p>那么,就推导出了我们的优化目标： <code>残差平方和最小</code></p>
<p>在Machine Learning中都是通过最小化Loss来达到训练的目的, 所以这个残差平方和就是Loss,并把它称之为MSE；</p>
<h3 id="%E7%AE%80%E5%8D%95%E5%AF%BC%E6%95%B0%E7%9F%A5%E8%AF%86%E6%8E%A8%E5%AF%BC%E8%A7%A3%E6%9E%90%E8%A7%A3theta-xtx-1xty">简单导数知识推导解析解($\theta = (X^TX)^{-1}X^TY$)</h3>

</body>
</html>






