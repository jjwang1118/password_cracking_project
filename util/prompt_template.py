import json
def _get_indice(id):
    if id == 0:
        return ("As a targeted password guessing model, your task is to utilize the provided account information to guess the password.")
    return ValueError


def prompt_convert(data:dict,prompt_template:str):

    # 轉換成文字prompt
    account = str(data.get("account", ""))
    password = data.get("password")
    if password is None:
        password = data.get("passwords", "")
    password = str(password)
    knowledge = f'{{"Username": "{account}"}}'
    prompt=prompt_template+knowledge+password
    return prompt
