from datasets import load_dataset

PATH="open-r1/OpenR1-Math-220k"
dataset = load_dataset(PATH)


def filter_correct(row):
    correctness_math_verify = row['correctness_math_verify']
    is_reasoning_complete = row['is_reasoning_complete']
    # print(len(correctness_math_verify))
    if len(correctness_math_verify) == 1:
        correctness_math_verify.append(False)
        is_reasoning_complete
    if len(correctness_math_verify) == 0:
        correctness_math_verify.append(False)
        correctness_math_verify.append(False)

        is_reasoning_complete.append(False)
        is_reasoning_complete.append(False)
    generations = row['generations']
    if correctness_math_verify[0]==True and is_reasoning_complete[0]==True:
        return {
            "correct_generation": generations[0],
        }
    elif correctness_math_verify[1]==True and is_reasoning_complete[1]==True:
        return {
            "correct_generation": generations[1],
        }
    else:
        return {
            "correct_generation": None,
        }
dataset = dataset['default']
dataset  = dataset.map(filter_correct, num_proc=16)
print(len(dataset))
dataset = dataset.filter(lambda x: x['correct_generation'] is not None)
print(len(dataset))
print(dataset[0]['correct_generation'])


def convert_to_chatml(row):
    problem = row['problem']
    system = "You are a the most powerful math expert. Please solve the problems with deep resoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. Please reason step by step, and put your final answer within \\boxed{}."
    user = f"{system}\n{problem}\n\n"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": row['correct_generation']#.replace("<think>", "").replace("</think>", "")
            }
        ]
    }

dataset = dataset.map(convert_to_chatml, num_proc=16)
keep_cols = ["problem", 'messages']
remove_cols = [col for col in dataset.column_names if col not in keep_cols]
dataset = dataset.remove_columns(remove_cols)
dataset = dataset.train_test_split(test_size=500)
print(len(dataset['train']))
print(len(dataset['test']))
dataset.push_to_hub("tuenguyen/open-r1-math-220k-chatml-v2")


