import gradio as gr

def greet(name):
    return "Greetings young man " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch(share=True)