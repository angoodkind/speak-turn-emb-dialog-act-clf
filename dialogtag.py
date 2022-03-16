import pandas as pd
from dialog_tag import DialogTag

## only load Dialogue Act model once
print('Loading DA model')
model = DialogTag('distilbert-base-uncased') # can use bert-base, but this is faster
print('DA model loaded')

messages = pd.read_csv('data/my_thesis/final_sends_dat.csv', sep='|', index_col=False)
messages = messages.reset_index()  # make sure indexes pair with number of rows

for index, row in messages.iterrows():
    predicted_da = model.predict_tag(row['sent_text'])
    messages.at[index,'dialogue_act'] = predicted_da
    # percent_done = round(index / len(messages.index), 2)
    if index % 100 == 0:
        print(index)
        # print(row)

messages.to_csv('sends_plus_da.csv', sep='|', index=False)