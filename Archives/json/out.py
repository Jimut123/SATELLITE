data = {'a':12}
import json
with open('file.json', 'a') as outfile:
    outfile.write(json.dumps(data))
    outfile.write(",")
    outfile.close()


