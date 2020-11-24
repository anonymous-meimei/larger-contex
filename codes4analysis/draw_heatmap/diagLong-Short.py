import string
import sys



def extValue(cont, fr, to):
    return cont.split(fr)[-1].split(to)[0] 




out_info = []
dict_att2pair = {}
deli = "####################################################"

all_cont =  sys.stdin.read()
for block in all_cont.split(deli):
    if block.find("____select_buckets____") == -1:
        continue
    corpus_type = extValue(block, "corpus_type: ", "\n")
    for inx, sec in enumerate(block.split("metric_names: ")):
        if inx == 0:
            continue
        attribute = sec.split("\n")[0]
        detas_subs = extValue(sec, "detas_subs: ", "\n")
        detas_subs = detas_subs.replace("[","").replace("]","").replace(", ","\t")
        #print type(detas_subs)
        print corpus_type, attribute, len(detas_subs.split("\t"))
        if dict_att2pair.has_key(attribute) == False:
            dict_att2pair[attribute] = [(corpus_type, detas_subs)]
        else:
            dict_att2pair[attribute].append((corpus_type, detas_subs))


def list2str(obj, sep):
    info = ""
    for val in obj:
        info = info + str(val) + sep
    return info.rstrip(sep)

def mat2str(obj):
    info = ""
    for row in obj:
        info = info + list2str(row) + "\n"
    return info.rstrip("\n")



path_base = "./heatmap/"
for k, v in dict_att2pair.iteritems():
    row_name_list = []
    vec_list = []
    n_col = 0
    fin = open(path_base+k+".csv","w")
    for dataset, vec in v:
        #if dataset in set(["notewb","wnut16"]):
        if dataset in set(["wnut16"]):
            continue
        dataset = dataset.replace("note","")
        dataset = dataset.replace("connl03","co")
        row_name_list.append(dataset)
        vec_list.append(vec)
        n_col = len(vec.split("\t"))

    fin.write(list2str(row_name_list,"\t")+"\n")
    #fin.write(list2str(range(n_col), "\t")+"\n")
    fin.write("XS\tS\tL\tXL\n")
    fin.write(list2str(vec_list,"\n")+"\n")
    fin.close()
