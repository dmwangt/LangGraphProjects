from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

response_schemas = [
    ResponseSchema(name="name", description="The name of the person"),
    ResponseSchema(name="age", description="The age of the person"),
    ResponseSchema(name="city", description="The city the person lives in"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template(
    """
    Generate a JSON object containing the person information.
    {format_instructions}
    person information: {person_information}
    """
)


# Initialize the chat model (GPT-4 variant or mini model for demonstration)
model = init_chat_model(
    "gemini-2.0-flash", model_provider="google_genai", temperature=0
)


person_information = "John, 30 years old, from New York."

messages = prompt.format_prompt(
    person_information=person_information,
    format_instructions=format_instructions,
).to_messages()

output = model.invoke(messages)
res = output_parser.parse(str(output.content))
print(res["name"])
print(res["age"])
print(res["city"])

print(res)
# Output:
# {
#     "name": "John",
#     "age": 30,
#     "city": "New York"
# }

## ---- list of string output ----
# There are many parsers available in langchain, including:
# - ListOutputParser
# - MarkdownListOutputParser
# - JsonOutputParser
# - JsonListOutputParser
# - StructuredOutputParser
# ...
from langchain.output_parsers import ListOutputParser, MarkdownListOutputParser
from langchain.prompts import PromptTemplate


output_parser_md = MarkdownListOutputParser()
template = """List any {num_people} people you know: {subject}\n{format_instructions}"""
prompt_template = PromptTemplate(
    template=template,
    input_variables=["num_people", "subject"],
    partial_variables={
        "format_instructions": output_parser_md.get_format_instructions()
    },
)

# Initialize the chat model (GPT-4 variant or mini model for demonstration)
model = init_chat_model(
    "gemini-2.0-flash", model_provider="google_genai", temperature=0
)

chain = prompt_template | model | output_parser_md

# so the output2 is a list of strings of python type
output2 = chain.invoke({"num_people": "3", "subject": "famous singers"})
for val in output2:
    print(val)
print("----")
print(output2)
