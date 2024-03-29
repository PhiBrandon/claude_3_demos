from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pandas as pd
from typing import List
from langfuse import Langfuse
from langfuse.client import StatefulTraceClient


load_dotenv()

anthropic = Anthropic()
langfuse = Langfuse()


class UserDetail(BaseModel):
    """
    Details about the user in the string.
    """

    name: str
    age: int
    occupation: str


class Skill(BaseModel):
    skill_label: str = Field(..., description="Label of skill, i.e. Soft skill, technical, etc...")
    skill_name: str
    skill_experience: str = Field(..., description="Years of experience mentioned for this skill, if it exists.")


class JobSkills(BaseModel):
    """Skills required for a job."""

    skills: List[Skill]
    job_description_summary: str



data_jobs = pd.read_csv("../../data/dataset_indeed-scraper_2023-10-23_07-07-15.csv")
job_desc = data_jobs["description"]

data = [
    "Samantha is 28 and a marketing analyst.",
    "Miguel is 35 and a software developer.",
    "Emily is 27 and a graphic designer.",
    "David is 41 and a project manager.",
    "Sophia is 23 and a social media specialist.",
    "Daniel is 33 and a financial advisor.",
    "Olivia is 26 and a HR coordinator.",
    "William is 39 and a civil engineer.",
    "Emma is 24 and a copywriter.",
    "Michael is 37 and a sales director.",
    "Isabella is 29 and a web developer.",
    "Jacob is 32 and a data scientist.",
    "Ava is 25 and a event planner.",
    "Matthew is 42 and a business consultant.",
    "Mia is 27 and a interior designer.",
    "Andrew is 34 and a IT manager.",
    "Abigail is 31 and a accountant.",
    "Joseph is 38 and a sales representative.",
    "Harper is 26 and a digital marketer.",
    "Joshua is 29 and a operations analyst.",
    "Amelia is 24 and a content creator.",
    "Christopher is 36 and a project engineer.",
    "Avery is 28 and a UX designer.",
    "Nicholas is 33 and a real estate agent.",
    "Evelyn is 30 and a paralegal.",
    "Tyler is 27 and a videographer.",
    "Chloe is 25 and a social worker.",
    "Caleb is 35 and a electrical engineer.",
    "Aubrey is 29 and a nutritionist.",
    "Ryan is 31 and a supply chain specialist.",
    "Charlotte is 26 and a public relations coordinator.",
    "Liam is 38 and a data architect.",
    "Leah is 24 and a customer service representative.",
    "Mason is 32 and a product designer.",
    "Naomi is 28 and a recruiter.",
    "Ethan is 37 and a logistics manager.",
    "Maya is 30 and a art director.",
    "Jacob is 25 and a teacher.",
    "Aria is 27 and a content strategist.",
    "Logan is 34 and a security analyst.",
    "Riley is 29 and a brand manager.",
    "Lucas is 36 and a database administrator.",
    "Zoe is 31 and a executive assistant.",
    "Michael is 40 and a construction manager.",
    "Lily is 26 and a translator.",
    "Alexander is 33 and a economist.",
    "Aubrey is 28 and a copywriter.",
    "Henry is 35 and a mechanical engineer.",
    "Avery is 24 and a social media manager.",
    "Benjamin is 29 and a business analyst.",
    "Scarlett is 27 and a market research analyst.",
    "Julian is 32 and a network engineer.",
    "Grace is 30 and a technical writer.",
    "Oliver is 38 and a management consultant.",
    "Chloe is 25 and a SEO specialist.",
    "James is 31 and a data analyst.",
    "Penelope is 26 and a public relations specialist.",
    "Levi is 34 and a solutions architect.",
    "Layla is 28 and a photographer.",
    "Sebastian is 36 and a cloud engineer.",
    "Brooklyn is 23 and a flight attendant.",
    "Samuel is 29 and a web designer.",
    "Madelyn is 27 and a program manager.",
    "Joseph is 33 and a industrial engineer.",
    "Reagan is 31 and a talent acquisition specialist.",
    "Gabriel is 35 and a DevOps engineer.",
    "Skylar is 26 and a front-end developer.",
    "Mateo is 30 and a business development manager.",
    "Ellie is 24 and a graphic artist.",
    "Dylan is 37 and a cybersecurity analyst.",
    "Lexi is 28 and a technical recruiter.",
    "Jeremiah is 32 and a systems administrator.",
    "Teagan is 25 and a digital analyst.",
    "Aaron is 39 and a production manager.",
    "Willow is 27 and a media buyer.",
    "Christian is 33 and a marketing manager.",
    "Athena is 29 and a research associate.",
    "Xavier is 35 and a solutions engineer.",
    "Luna is 26 and a motion graphics designer.",
    "Nguyễn Thị Lan is 28 and a marketing analyst.",
    "Trần Minh Hoàng is 35 and a software developer.",
    "Lê Thị Mai is 27 and a graphic designer.",
    "Phạm Văn Dũng is 41 and a project manager.",
    "Vũ Thiên Hương is 23 and a social media specialist.",
    "Đặng Quốc Anh is 33 and a financial advisor.",
    "Trần Thị Hồng is 26 and a HR coordinator.",
    "Ngô Văn Hùng is 39 and a civil engineer.",
    "Hoàng Thị Thảo is 24 and a copywriter.",
    "Lê Quang Vinh is 37 and a sales director.",
    "Đỗ Bích Ngọc is 29 and a web developer.",
    "Võ Minh Khôi is 32 and a data scientist.",
    "Trần Thị Nhung is 25 and a event planner.",
    "Phạm Văn Lộc is 42 and a business consultant.",
    "Nguyễn Thị Ngân is 27 and a interior designer.",
    "Đoàn Quốc Bảo is 34 and a IT manager.",
    "Lê Thị Tuyết is 31 and a accountant.",
    "Hoàng Văn Dũng is 38 and a sales representative.",
    "Vũ Thị Hồng is 26 and a digital marketer.",
    "Trần Minh Khải is 29 and a operations analyst.",
    "Đỗ Thị Huyền is 24 and a content creator.",
    "Lê Quang Trung is 36 and a project engineer.",
    "Phạm Thị Lan Anh is 28 and a UX designer.",
    "Nguyễn Văn Quốc is 33 and a real estate agent.",
    "Vũ Thị Huyền is 30 and a paralegal.",
    "Đặng Minh Tuấn is 27 and a videographer.",
    "Lê Thị Hà is 25 and a social worker.",
    "Phạm Xuân Hùng is 35 and a electrical engineer.",
    "Trần Thị Thu is 29 and a nutritionist.",
    "Hoàng Văn Đạt is 31 and a supply chain specialist.",
    "Ngô Thị Khánh is 26 and a public relations coordinator.",
    "Đỗ Minh Quân is 38 and a data architect.",
    "Võ Thị Ngọc is 24 and a customer service representative.",
    "Trần Văn Phong is 32 and a product designer.",
    "Lê Thị Linh is 28 and a recruiter.",
    "Nguyễn Công Sơn is 37 and a logistics manager.",
    "Đặng Thị Hồng Hạnh is 30 and a art director.",
    "Vũ Hữu Lộc is 25 and a teacher.",
    "Phạm Thị Ngọc Diệp is 27 and a content strategist.",
    "Lê Quang Huy is 34 and a security analyst.",
    "Trần Thị Diễm is 29 and a brand manager.",
    "Đỗ Văn Tiến is 36 and a database administrator.",
    "Nguyễn Thị Hương is 31 and a executive assistant.",
    "Hoàng Quốc Tuấn is 40 and a construction manager.",
    "Vũ Thị Thu Hà is 26 and a translator.",
    "Trần Minh Quang is 33 and a economist.",
    "Lê Thị Thúy is 28 and a copywriter.",
    "Phạm Ngọc Hưng is 35 and a mechanical engineer.",
    "Đỗ Thị Hảo is 24 and a social media manager.",
    "Nguyễn Văn Đức is 29 and a business analyst.",
    "Vũ Thị Diễm is 27 and a market research analyst.",
    "Lê Quang Thịnh is 32 and a network engineer.",
    "Trần Thị Cẩm Tú is 30 and a technical writer.",
    "Đặng Văn Nam is 38 and a management consultant.",
    "Phạm Thị Lan Anh is 25 and a SEO specialist.",
    "Hoàng Quốc Hùng is 31 and a data analyst.",
    "Ngô Thị Thanh is 26 and a public relations specialist.",
    "Vũ Minh Khôi is 34 and a solutions architect.",
    "Lê Thị Quỳnh Trang is 28 and a photographer.",
    "Trần Văn Sơn is 36 and a cloud engineer.",
    "Đỗ Thị Thu Huyền is 23 and a flight attendant.",
    "Nguyễn Công Hiếu is 29 and a web designer.",
    "Vũ Thị Tú Quyên is 27 and a program manager.",
    "Phạm Minh Đức is 33 and a industrial engineer.",
    "Lê Thị Thu Thảo is 31 and a talent acquisition specialist.",
    "Đặng Quốc Tuấn is 35 and a DevOps engineer.",
    "Trần Thị Huyền Trâm is 26 and a front-end developer.",
    "Hoàng Văn Quyền is 30 and a business development manager.",
    "Nguyễn Thị Thư is 24 and a graphic artist.",
    "Lê Quang Trí is 37 and a cybersecurity analyst.",
    "Phạm Thị Hải Yến is 28 and a technical recruiter.",
    "Đỗ Minh Đức is 32 and a systems administrator.",
    "Vũ Thị Hồng Nhung is 25 and a digital analyst.",
    "Trần Văn Huy is 39 and a production manager.",
    "Lê Thị Mỹ Linh is 27 and a media buyer.",
    "Nguyễn Quốc Hiếu is 33 and a marketing manager.",
    "Đặng Thị Ánh is 29 and a research associate.",
    "Phạm Văn Quân is 35 and a solutions engineer.",
    "Vũ Thị Thảo Ly is 26 and a motion graphics designer.",
    "María Rodríguez is 28 and a marketing analyst.",
    "José Gómez is 35 and a software developer.",
    "Ana Hernández is 27 and a graphic designer.",
    "Carlos Torres is 41 and a project manager.",
    "Lucía García is 23 and a social media specialist.",
    "Miguel Martínez is 33 and a financial advisor.",
    "Isabela Ramírez is 26 and a HR coordinator.",
    "Diego Flores is 39 and a civil engineer.",
    "Valentina Pérez is 24 and a copywriter.",
    "Alejandro Sánchez is 37 and a sales director.",
    "Sofía Castillo is 29 and a web developer.",
    "Mateo Cabrera is 32 and a data scientist.",
    "Camila Gutiérrez is 25 and a event planner.",
    "Juan Morales is 42 and a business consultant.",
    "Renata Rivera is 27 and a interior designer.",
    "Andrés Romero is 34 and a IT manager.",
    "Mariana Herrera is 31 and a accountant.",
    "Jorge Díaz is 38 and a sales representative.",
    "Valeria Medina is 26 and a digital marketer.",
    "Santiago Núñez is 29 and a operations analyst.",
    "Daniela Vázquez is 24 and a content creator.",
    "Gabriel Rojas is 36 and a project engineer.",
    "Emilia Castro is 28 and a UX designer.",
    "Ricardo Salazar is 33 and a real estate agent.",
    "Fernanda Espinoza is 30 and a paralegal.",
    "Samuel Ríos is 27 and a videographer.",
    "Mía Ramos is 25 and a social worker.",
    "Javier Reyes is 35 and a electrical engineer.",
    "Abril Fuentes is 29 and a nutritionist.",
    "Diego Mendoza is 31 and a supply chain specialist.",
    "Valeria Contreras is 26 and a public relations coordinator.",
    "Manuel Guerrero is 38 and a data architect.",
    "Sara Arellano is 24 and a customer service representative.",
    "Emiliano Cruz is 32 and a product designer.",
    "Luciana Muñoz is 28 and a recruiter.",
    "David Aguilar is 37 and a logistics manager.",
    "Catalina Figueroa is 30 and a art director.",
    "Matías Navarro is 25 and a teacher.",
    "Isabella Sandoval is 27 and a content strategist.",
    "Lucas Jiménez is 34 and a security analyst.",
    "Valentina Delgado is 29 and a brand manager.",
    "Benjamín Ortega is 36 and a database administrator.",
    "Sofía Huerta is 31 and a executive assistant.",
    "Alejandro Ponce is 40 and a construction manager.",
    "Renata Rivas is 26 and a translator.",
    "Javier Lagos is 33 and a economist.",
    "Camila Salas is 28 and a copywriter.",
    "Andrés Henríquez is 35 and a mechanical engineer.",
    "Valeria Acuña is 24 and a social media manager.",
    "Santiago Narváez is 29 and a business analyst.",
    "Daniela Padilla is 27 and a market research analyst.",
    "Gabriel Quintero is 32 and a network engineer.",
    "Sara Mayorga is 30 and a technical writer.",
    "Emiliano Vega is 38 and a management consultant.",
    "Lucía Cordero is 25 and a SEO specialist.",
    "Diego Cáceres is 31 and a data analyst.",
    "Fernanda Tapia is 26 and a public relations specialist.",
    "Samuel Montero is 34 and a solutions architect.",
    "Mía Peralta is 28 and a photographer.",
    "Javier Calderón is 36 and a cloud engineer.",
    "Catalina Farías is 23 and a flight attendant.",
    "Mateo Soto is 29 and a web designer.",
    "Daniela Molina is 27 and a program manager.",
    "Carlos Campos is 33 and a industrial engineer.",
    "Sara Zúñiga is 31 and a talent acquisition specialist.",
    "David Guzmán is 35 and a DevOps engineer.",
    "Luciana Araneda is 26 and a front-end developer.",
    "Lucas Moreira is 30 and a business development manager.",
    "Sofía Ávila is 24 and a graphic artist.",
    "Alejandro Pizarro is 37 and a cybersecurity analyst.",
    "Camila Alarcón is 28 and a technical recruiter.",
    "Emiliano Parra is 32 and a systems administrator.",
    "Valentina Lara is 25 and a digital analyst.",
    "Samuel Piña is 39 and a production manager.",
    "Renata Leiva is 27 and a media buyer.",
    "Mateo Céspedes is 33 and a marketing manager.",
    "Mía Gallardo is 29 and a research associate.",
    "Javier Balmaceda is 35 and a solutions engineer.",
    "Catalina Ibáñez is 26 and a motion graphics designer.",
    "Jamal Williams is 28 and a marketing analyst.",
    "Tiffany Jackson is 35 and a software developer.",
    "Darnell Harris is 27 and a graphic designer.",
    "Latisha Thompson is 41 and a project manager.",
    "Terrell Johnson is 23 and a social media specialist.",
    "Tamika Brown is 33 and a financial advisor.",
    "Darius Robinson is 26 and a HR coordinator.",
    "Keisha Davis is 39 and a civil engineer.",
    "Malik Wilson is 24 and a copywriter.",
    "Shanice Taylor is 37 and a sales director.",
    "Jermaine Anderson is 29 and a web developer.",
    "Aaliyah Moore is 32 and a data scientist.",
    "Kwame Lee is 25 and a event planner.",
    "Latoya Martin is 42 and a business consultant.",
    "Rasheed Walker is 27 and a interior designer.",
    "Terrell Thomas is 34 and a IT manager.",
    "Ebony Washington is 31 and a accountant.",
    "Demetrius Green is 38 and a sales representative.",
    "Tameka Jones is 26 and a digital marketer.",
    "Jamal Roberts is 29 and a operations analyst.",
    "Jasmine Edwards is 24 and a content creator.",
    "Jermaine Lewis is 36 and a project engineer.",
    "Tia Young is 28 and a UX designer.",
    "Darius Hill is 33 and a real estate agent.",
    "Jasmine King is 30 and a paralegal.",
    "Terrence Ross is 27 and a videographer.",
    "Aaliyah Scott is 25 and a social worker.",
    "Kwame Mitchell is 35 and a electrical engineer.",
    "Ebony Phillips is 29 and a nutritionist.",
    "Jermaine Parker is 31 and a supply chain specialist.",
    "Tameka Graham is 26 and a public relations coordinator.",
    "Jamal Bryant is 38 and a data architect.",
    "Jasmine Richardson is 24 and a customer service representative.",
    "Malik Hudson is 32 and a product designer.",
    "Latisha Watson is 28 and a recruiter.",
    "Terrell Bell is 37 and a logistics manager.",
    "Tiffany Morgan is 30 and a art director.",
    "Darnell Barnes is 25 and a teacher.",
    "Keisha Ellis is 27 and a content strategist.",
    "Malik Foster is 34 and a security analyst.",
    "Tamika Murray is 29 and a brand manager.",
    "Darius Powell is 36 and a database administrator.",
    "Ebony Cooper is 31 and a executive assistant.",
    "Terrell Perry is 40 and a construction manager.",
    "Jasmine Baker is 26 and a translator.",
    "Jermaine Curry is 33 and a economist.",
    "Latoya Henderson is 28 and a copywriter.",
    "Kwame Wilkins is 35 and a mechanical engineer.",
    "Tameka Murphy is 24 and a social media manager.",
    "Malik Allen is 29 and a business analyst.",
    "Keisha Nelson is 27 and a market research analyst.",
    "Jamal Simmons is 32 and a network engineer.",
    "Tiffany Patterson is 30 and a technical writer.",
    "Darnell Hawkins is 38 and a management consultant.",
    "Terrell Griffin is 25 and a SEO specialist.",
    "Latisha Brooks is 31 and a data analyst.",
    "Tamika Stewart is 26 and a public relations specialist.",
    "Darius Franklin is 34 and a solutions architect.",
    "Jasmine Harper is 28 and a photographer.",
    "Malik Patel is 36 and a cloud engineer.",
    "Keisha Jennings is 23 and a flight attendant.",
    "Terrell Summers is 29 and a web designer.",
    "Ebony Evans is 27 and a program manager.",
    "Jermaine Mcdonald is 33 and a industrial engineer.",
    "Latoya Rhodes is 31 and a talent acquisition specialist.",
    "Malik Carpenter is 35 and a DevOps engineer.",
    "Tamika Butler is 26 and a front-end developer.",
    "Darnell Underwood is 30 and a business development manager.",
    "Jasmine Gibbs is 24 and a graphic artist.",
    "Terrence Ford is 37 and a cybersecurity analyst.",
    "Tiffany Grant is 28 and a technical recruiter.",
    "Jamal Lane is 32 and a systems administrator.",
    "Latisha Sanders is 25 and a digital analyst.",
    "Darius Watkins is 39 and a production manager.",
    "Keisha Moss is 27 and a media buyer.",
    "Malik Thornton is 33 and a marketing manager.",
    "Tameka Ferguson is 29 and a research associate.",
    "Jermaine Nguyen is 35 and a solutions engineer.",
    "Ebony Cross is 26 and a motion graphics designer.",
    "Wang Xiaoming is 28 and a marketing analyst.",
    "Li Yingying is 35 and a software developer.",
    "Zhang Mei is 27 and a graphic designer.",
    "Liu Zhiyuan is 41 and a project manager.",
    "Chen Xiaoxue is 23 and a social media specialist.",
    "Huang Zhenyu is 33 and a financial advisor.",
    "Wu Xinyi is 26 and a HR coordinator.",
    "Zhou Haoran is 39 and a civil engineer.",
    "Zhu Yilin is 24 and a copywriter.",
    "Sun Jinming is 37 and a sales director.",
    "Yang Xueying is 29 and a web developer.",
    "Hu Zhixiao is 32 and a data scientist.",
    "Guo Xinran is 25 and a event planner.",
    "Lin Yifan is 42 and a business consultant.",
    "He Meiling is 27 and a interior designer.",
    "Zheng Zitao is 34 and a IT manager.",
    "Xu Haiyan is 31 and a accountant.",
    "Qian Jiahao is 38 and a sales representative.",
    "Cheng Bingbing is 26 and a digital marketer.",
    "Jiang Zixuan is 29 and a operations analyst.",
    "Shen Ruoxi is 24 and a content creator.",
    "Tan Yuxuan is 36 and a project engineer.",
    "Liang Jiaying is 28 and a UX designer.",
    "Wang Ziheng is 33 and a real estate agent.",
    "Zhou Meilin is 30 and a paralegal.",
    "Chen Siming is 27 and a videographer.",
    "Liu Jiaying is 25 and a social worker.",
    "Huang Yuechen is 35 and a electrical engineer.",
    "Wu Qianyi is 29 and a nutritionist.",
    "Zhang Xingyu is 31 and a supply chain specialist.",
    "Zhu Jiaxin is 26 and a public relations coordinator.",
    "Sun Ziyang is 38 and a data architect.",
    "Yang Xinchen is 24 and a customer service representative.",
    "Hu Yihan is 32 and a product designer.",
    "Guo Zixin is 28 and a recruiter.",
    "Lin Chengxin is 37 and a logistics manager.",
    "He Yuhan is 30 and a art director.",
    "Zheng Qingxuan is 25 and a teacher.",
    "Xu Jiaying is 27 and a content strategist.",
    "Qian Zixuan is 34 and a security analyst.",
    "Cheng Yunxi is 29 and a brand manager.",
    "Jiang Hanchen is 36 and a database administrator.",
    "Shen Yingying is 31 and a executive assistant.",
    "Tan Haoran is 40 and a construction manager.",
    "Liang Xinyi is 26 and a translator.",
    "Wang Jiahao is 33 and a economist.",
    "Zhou Xueer is 28 and a copywriter.",
    "Chen Mingyu is 35 and a mechanical engineer.",
    "Liu Qianyi is 24 and a social media manager.",
    "Huang Zixun is 29 and a business analyst.",
    "Wu Xinru is 27 and a market research analyst.",
    "Zhang Yichen is 32 and a network engineer.",
    "Zhu Siyuan is 30 and a technical writer.",
    "Sun Zhiyu is 38 and a management consultant.",
    "Yang Qianyun is 25 and a SEO specialist.",
    "Hu Shihan is 31 and a data analyst.",
    "Guo Yaqi is 26 and a public relations specialist.",
    "Lin Jiahao is 34 and a solutions architect.",
    "He Xinru is 28 and a photographer.",
    "Zheng Chengxu is 36 and a cloud engineer.",
    "Xu Qianqian is 23 and a flight attendant.",
    "Qian Yuhan is 29 and a web designer.",
    "Cheng Ziying is 27 and a program manager.",
    "Jiang Junhao is 33 and a industrial engineer.",
    "Shen Jiayi is 31 and a talent acquisition specialist.",
    "Tan Zhenhuan is 35 and a DevOps engineer.",
    "Liang Yuxin is 26 and a front-end developer.",
    "Wang Xingchen is 30 and a business development manager.",
    "Zhou Qingyi is 24 and a graphic artist.",
    "Chen Zixin is 37 and a cybersecurity analyst.",
    "Liu Mingxuan is 28 and a technical recruiter.",
    "Huang Jiachen is 32 and a systems administrator.",
    "Wu Xinrui is 25 and a digital analyst.",
    "Zhang Yuchen is 39 and a production manager.",
    "Zhu Yumeng is 27 and a media buyer.",
    "Sun Zhiwei is 33 and a marketing manager.",
    "Yang Muyan is 29 and a research associate.",
    "Hu Zixuan is 35 and a solutions engineer.",
    "Guo Yihan is 26 and a motion graphics designer.",
]


def get_user_detail(
    data: str, pydantic_model: BaseModel, trace: StatefulTraceClient, version: str
):
    system_prompt = """Here's a JSON schema to follow: {pydantic_schema} Output a valid JSON object but do not repeat the schema."""
    user_prompt = f"Extract the user detail from the following text: {data}"
    schema = pydantic_model.model_json_schema()

    trace_generation = trace.generation(
        name="user_detail_claude_haiku",
        model="claude-3-haiku-20240307",
        input={"system_prompt": system_prompt, "pydantic_schema": schema, "data": data},
        version=version,
        usage={
            # usage
            "input": int,
            "output": int,
            "total": int,  # if not set, it is derived from input + output
            "unit": "TOKENS",  # any of: "TOKENS", "CHARACTERS", "MILLISECONDS", "SECONDS", "IMAGES"
            # usd cost
            "input_cost": float,
            "output_cost": float,
            "total_cost": float,  # if not set, it is derived from input_cost + output_cost
        },
    )

    output = anthropic.messages.create(
        model="claude-3-haiku-20240307",
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        max_tokens=1000,
        system=system_prompt.format(pydantic_schema=schema),
        temperature=0,
    )

    try:
        valid_user = pydantic_model.model_validate_json(output.content[0].text)
        total = (output.usage.input_tokens * 0.00000025) + (
            output.usage.output_tokens * 0.00000124
        )
        trace_generation.end(
            output={"raw": output.content[0].text, "pydantic_output": valid_user},
            status_message="Success",
            model="claude-3-haiku-20240307",
            usage={
                # usage
                "input": output.usage.input_tokens,
                "output": output.usage.output_tokens,
                "unit": "TOKENS",
                # usd cost
                "input_cost": output.usage.input_tokens * 0.00000025,
                "output_cost": output.usage.output_tokens * 0.00000124,
                "total_cost": total,
            },
        )
        print(valid_user)
    except Exception as e:
        total = (output.usage.input_tokens * 0.00000025) + (
            output.usage.output_tokens * 0.00000124
        )
        trace_generation.end(
            output={"raw": output.content[0].text, "error_output": e},
            status_message="Error",
            usage={
                # usage
                "input": output.usage.input_tokens,
                "output": output.usage.output_tokens,
                "unit": "TOKENS",
                # usd cost
                "input_cost": output.usage.input_tokens * 0.00000025,
                "output_cost": output.usage.output_tokens * 0.00000124,
                "total_cost": total,
            },
        )
        print(f"Raw: {output.content[0].text}")
        print(f"Error: {e}")


def get_job_skills(
    data: str, pydantic_model: BaseModel, trace: StatefulTraceClient, version: str, trace_name: str
):
    system_prompt = """Here is a pydantic schema for a JSON object:

                    <pydantic_schema>
                    {pydantic_schema}
                    </pydantic_schema>

                    Please study this schema carefully. Your task is to output a valid JSON object that follows this
                    schema exactly. Do not include the schema itself in your output, only the JSON object.

                    Do not include any other text or explanation in your
                    response"""
    schema = pydantic_model.model_json_schema()
    user_prompt = f"Extract neccessary skills from the job description: <job_description>{data}</job_description>"
    trace_generation = trace.generation(
        name=trace_name,
        model="claude-3-haiku-20240307",
        input={
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "pydantic_schema": schema,
            "data": data,
        },
        version=version,
        usage={
            # usage
            "input": int,
            "output": int,
            "total": int,  # if not set, it is derived from input + output
            "unit": "TOKENS",  # any of: "TOKENS", "CHARACTERS", "MILLISECONDS", "SECONDS", "IMAGES"
            # usd cost
            "input_cost": float,
            "output_cost": float,
            "total_cost": float,  # if not set, it is derived from input_cost + output_cost
        },
    )
    output = anthropic.messages.create(
        model="claude-3-haiku-20240307",
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        max_tokens=1000,
        system=system_prompt.format(pydantic_schema=schema),
        temperature=0,
    )

    try:
        valid_user = pydantic_model.model_validate_json(output.content[0].text)
        total = (output.usage.input_tokens * 0.00000025) + (
            output.usage.output_tokens * 0.00000124
        )
        trace_generation.end(
            output={"raw": output.content[0].text, "pydantic_output": valid_user},
            status_message="Success",
            model="claude-3-haiku-20240307",
            usage={
                # usage
                "input": output.usage.input_tokens,
                "output": output.usage.output_tokens,
                "unit": "TOKENS",
                # usd cost
                "input_cost": output.usage.input_tokens * 0.00000025,
                "output_cost": output.usage.output_tokens * 0.00000124,
                "total_cost": total,
            },
        )
        print(valid_user)
    except Exception as e:
        total = (output.usage.input_tokens * 0.00000025) + (
            output.usage.output_tokens * 0.00000124
        )
        trace_generation.end(
            output={"raw": output.content[0].text, "error_output": e},
            status_message="Error",
            usage={
                # usage
                "input": output.usage.input_tokens,
                "output": output.usage.output_tokens,
                "unit": "TOKENS",
                # usd cost
                "input_cost": output.usage.input_tokens * 0.00000025,
                "output_cost": output.usage.output_tokens * 0.00000124,
                "total_cost": total,
            },
        )
        print(f"Raw: {output.content[0].text}")
        print(f"Error: {e}")


version = "0.0.1"
user_detai_trace = langfuse.trace(name="user_detail_claude", version=version)

for d in data:
    get_user_detail(d, UserDetail, user_detai_trace, version)

version = "0.0.4"
trace_name = "job_skills_claude_haiku"
job_skills_trace = langfuse.trace(name=trace_name, version=version)
for j in job_desc:
    get_job_skills(j, JobSkills, job_skills_trace, version, trace_name)

