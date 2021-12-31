import json
import glob
from pathlib import Path
import shutil

def main():
    output = 'data/training_data/'
    Path(output).mkdir(parents=True, exist_ok=True) 

    for gt_file in glob.iglob('data/meeter_annotations/*.json'):
      print(gt_file)
      img_file = gt_file.replace('.json', '.jpeg')
      gt =  json.load(open(gt_file))

      with open(output + Path(gt_file).stem + '.txt', 'w') as f:
        for poly in gt['shapes']:
          if poly['label'] == 'meeter-reading':
            line = ''
            for point in poly['points']:
              line += ','.join([str(int(x)) for x in point])
              line += ','
            f.write(line + 'abc\n')

        shutil.copy(img_file, output)

if __name__=="__main__":
    main()