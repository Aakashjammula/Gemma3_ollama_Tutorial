import base64
from io import BytesIO
import gradio as gr
from PIL import Image
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def convert_to_base64(pil_image):
    if pil_image.mode == "P" or pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB") 

    format = pil_image.format if pil_image.format else "PNG"

    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str, format.lower() 

def chat_with_model(user_message, image):
    model = ChatOllama(model="gemma3:4b", temperature=0.5)
    conversation_history = []

    image_b64 = None
    if image is not None:
        pil_image = Image.open(image)
        image_b64, _ = convert_to_base64(pil_image)

    message_content = [{"type": "text", "text": user_message}]
    if image_b64:
        message_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        )
    message = HumanMessage(content=message_content)
    conversation_history.append(message)
    
    ai_msg = model.invoke(conversation_history)
    conversation_history.append(ai_msg)
    
    return ai_msg.content  

if __name__ == "__main__":
    demo = gr.Interface(
        fn=chat_with_model,
        inputs=[gr.Textbox(label="Your Message"), gr.Image(type="filepath", label="Upload Image (Optional)")],
        outputs=gr.Textbox(label="AI Response"),
        title="Chat with Ollama",
        description="Interact with the AI model using text and optional images."
    )

    demo.launch()