import argparse as ap
import sort

parser = ap.ArgumentParser()
parser.add_argument("--file", type=str or int, default=1, help="Either int or str. If value is integer, it will search in scores.json the file of that position. Example: if value is 1, it will use the best model.")

config = parser.parse_args()

print(config.file, type(config.file))

file = config.file

if type(config.file) == int:    
    file = sort.best_model_get_name(1)
    print(1)
elif type(config.file) == str:  
    file = "Logs/{}".format(config.file)
    print(2)
else:                           
    print(3)
    print( "ERROR! argument '--file' is not str or int. The value must be one of those")
    exit()

print(file)