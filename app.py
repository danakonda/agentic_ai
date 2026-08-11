from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool
import pandas as pd
from datetime import datetime
load_dotenv()
from pydantic import BaseModel
from fastapi import FastAPI
import os
from pathlib import Path
from google import genai
from huggingface_hub import InferenceClient
import yt_dlp
import uuid
from fastapi.responses import FileResponse
image_client=InferenceClient(api_key=os.getenv("HF_API_KEY"))
output_dir=Path('outputs')
output_dir.mkdir(exist_ok=True)
llm = ChatGoogleGenerativeAI(api_key=os.getenv('API_KEY'),model='gemini-3.5-flash')
search_tool = TavilySearch(topic="general",search_depth="advanced")



app=FastAPI(title="AI agent API")
@tool
def write_csv(filename: str, data:list[dict]) ->str:
    """
    Creates a csv file using pandas .
    filename:
    Name of the csv file.
    data:
        list of dictionaries containing the rows.
    """

    if not filename.endswith(".csv"):
        filename +=".csv"
    file_path=output_dir/filename
    df=pd.DataFrame(data)
    df.to_csv(file_path,index=False)
    return f"csv file created successfully: {file_path}"

    
@tool
def get_datetime()->str:
    """
    Returns the current date and time.
    """
    now=datetime.now()
    return now.strftime("%d-%m-%Y %H:%M:%S")


@tool
def generate_image(
    prompt: str,
    filename: str = "generated_image.png"
) -> str:
    """
    Generate an AI image using Hugging Face FLUX.1-dev
    and save it as a PNG file.
    """

    try:
        if not filename.endswith(".png"):
            filename += ".png"

        file_path = output_dir / filename

        image = image_client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-dev"
        )

        image.save(file_path)

        return f"Image generated successfully: {file_path}"

    except Exception as e:
        return f"Image generation failed: {str(e)}"
@tool
def youtube_video(topic:str)->str:
    """
    search youtube for a video based on the given topic.
    the user only needs to provide a topic.
    the tool automatically finds a youtube video,extracts its video id ,creates the youtube url,and creates a local fastpai download url.
    """
    try:
        ydl_opts={
            "quiet":True,
            "extract_flat":True,
            "skip_download":True
        }
        query=f"ytsearch1:{topic}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result=ydl.extract_info(query,download=False)
            videos=result.get("entries",[])
            if not videos:
                return "NO youtube video found."
            video=videos[0]
            video_id=video.get('id')
            youtube_url=(video.get("webpage_url")or f"https://www.youtube.com/watch?v={video_id}")
            base_url=os.getenv("BASE_URL","http://127.0.0.1:8000")

            download_url=(f"{base_url}/download/{video_id}")

            return (
                f"Title:{video.get('title')}\n"
                f"channel:{video.get('channel','Unknown')}\n"
                f"Youtube url:{youtube_url}\n"
                f"download url:{download_url}"

            )
    except Exception as e:
        return f"youtube search failed:{str(e)}"

@app.get("/download/{video_id}")
def download_video(video_id:str):
    print("download request:", video_id)
    
    url=f"https://www.youtube.com/watch?v={video_id}"
    os.makedirs("downloads",exist_ok=True)
    filename=str(uuid.uuid4())
    output=f"downloads/{filename}.%(ext)s"
    ydl_opts={
        "format":"best[ext=mp4]/best",
        "outtmpl":output,
        "noplaylist":True

    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info=ydl.extract_info(url,download=True)
            filepath=ydl.prepare_filename(info)
        return FileResponse(path=filepath,filename=os.path.basename(filepath),media_type="application/octent-stream")
    except Exception as e:
        return {
            "error":str(e)

        }
spl_prompt = """You are an AI Agent that can make a decision whether or not to use
the search tool. Use the search tool only when the user asks about recent events .
For general responses, do not use the search tool.
1.get_datetime
  -gets the current date and time
2.write_csv 
  -creates csv file using pandas .
  -Use it whenever the user asks to create or save csv data.
3.generate_image
   -generates and saves ai images .
   -use it whenever the user asks to create ,generate ,or save an image.
4.youtube_video
  -use topics search url give the download option
  -automatically find a youtube video
  -return the youtube url.
  -return the download url.
  -do not ask user for video id if a topic is probided.

"""

agent = create_agent(model=llm, tools=[search_tool,write_csv,get_datetime,generate_image,youtube_video], system_prompt=spl_prompt)
class PromptRequest(BaseModel):
    prompt:str

@app.post("/chat")
def chat(req:PromptRequest):
    
    response = agent.invoke({'messages': [('human', req.prompt)]})
    last_message=response["messages"][-1]
    content=last_message.content
    if isinstance(content,str):
        result=content
    elif isinstance(content,list):
        texts=[]
        for item in content:
            if isinstance(item,dict)and "text" in item:
                texts.append(item["text"])
        result="\n".join(texts)
    else:
        result=str(content)

    return {
        "response":result
    }

    