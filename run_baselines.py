from recbole.quick_start import run_recbole

# dataset to url: https://github.com/RUCAIBox/RecBole/blob/master/recbole/properties/dataset/url.yaml

print("start")

# dataset='amazon-beauty'
# dataset='amazon-sports-outdoors'
dataset='amazon-toys-games'

# run_recbole(model='BERT4Rec', dataset=dataset, config_file_list=['./bert4rec.yaml'])
# run_recbole(model='LightGCN', dataset=dataset, config_file_list=['./lightgcn.yaml'])
run_recbole(model='SASRec', dataset=dataset, config_file_list=['./sasrec.yaml'])

