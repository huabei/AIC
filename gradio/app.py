import gradio as gr
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 浓度和吸光度数据
c = [0.01, 0.02, 0.03, 0.04, 0.05]  # 浓度
a = [0.14, 0.31, 0.46, 0.60, 0.73]  # 吸光度

# 建立线性模型
model = LinearRegression()
X = np.array(c).reshape(-1, 1)
y = np.array(a)
model.fit(X, y)

def predict_concentration(absorbance_str):
    """根据吸光度字符串预测浓度，并返回文本结果及数值"""
    try:
        absorbance_float = float(absorbance_str)
    except ValueError:
        return "错误：请输入有效的数字作为吸光度。", None, None
    
    predicted_conc_float = (absorbance_float - model.intercept_) / model.coef_[0]
    prediction_text = f"预测浓度: {predicted_conc_float:.5f}"
    return prediction_text, absorbance_float, predicted_conc_float

def plot_model(current_absorbance=None, current_concentration=None):
    """Plot concentration-absorbance relationship, linear model and current input point"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(c, a, color='blue', label='Experimental Data')
    
    c_line = np.linspace(min(c)*0.8, max(c)*1.2, 100) # Adjusted range for better display
    if current_concentration is not None: # Ensure plot range includes current point
        c_line = np.linspace(min(min(c)*0.8, current_concentration*0.8), max(max(c)*1.2, current_concentration*1.2), 100)

    a_line = model.predict(c_line.reshape(-1, 1))
    ax.plot(c_line, a_line, color='red', label=f'Fitted line: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
    
    if current_absorbance is not None and current_concentration is not None:
        ax.scatter([current_concentration], [current_absorbance], color='green', s=100, marker='*', label='Current Input/Prediction', zorder=5)

    ax.set_xlabel('Concentration (c)')
    ax.set_ylabel('Absorbance (a)')
    ax.set_title('Linear Relationship between Concentration and Absorbance')
    ax.grid(True)
    ax.legend()
    
    return fig

def generate_prediction_and_plot(absorbance_str):
    """生成预测文本和更新后的图表"""
    prediction_text, num_abs, num_conc = predict_concentration(absorbance_str)
    fig = plot_model(num_abs, num_conc)
    return prediction_text, fig

with gr.Blocks() as app:
    gr.Markdown("# 浓度-吸光度线性模型")
    gr.Markdown(f"线性拟合结果: a = {model.coef_[0]:.4f}c + {model.intercept_:.4f}, R² = {model.score(X, y):.4f}")

    with gr.Row():
        with gr.Column(scale=1):
            abs_input = gr.Textbox(label="输入吸光度", placeholder="例如: 0.5")
            pred_text_output = gr.Textbox(label="预测结果", interactive=False)
            submit_button = gr.Button("预测")
            gr.Markdown("输入一个吸光度值，模型将预测对应的浓度。")
            
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="模型图表")

    gr.Examples(
        examples=[["0.25"], ["0.50"], ["0.65"]],
        inputs=abs_input,
        outputs=[pred_text_output, plot_output],
        fn=generate_prediction_and_plot,
        cache_examples=False # Matplotlib图表可能不适合缓存
    )

    # 应用加载时显示初始图表
    app.load(lambda: plot_model(None, None), inputs=None, outputs=plot_output)

    # 事件处理
    submit_button.click(
        fn=generate_prediction_and_plot,
        inputs=abs_input,
        outputs=[pred_text_output, plot_output]
    )
    abs_input.submit(
        fn=generate_prediction_and_plot,
        inputs=abs_input,
        outputs=[pred_text_output, plot_output]
    )

app.launch(share=True)
