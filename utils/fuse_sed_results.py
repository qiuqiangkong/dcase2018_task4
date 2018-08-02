import csv
import os
import argparse

from utilities import create_folder


def fuse_sed_results(args):
    
    workspace = args.workspace
    sed1_path = args.sed1_path
    sed2_path = args.sed2_path
    
    out_path = os.path.join(workspace, 'submissions', 'fuse_sed_results', 'fused_sed.csv')
    create_folder(os.path.dirname(out_path))
    
    sed1_events = ['Frying', 'Blender', 'Running_water', 'Vacuum_cleaner', 'Electric_shaver_toothbrush']
    sed2_events = ['Speech', 'Dog', 'Cat', 'Alarm_bell_ringing', 'Dishes']
    
    new_list = []
    
    with open(sed1_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
        
        for li in lis:
            label = li[3]
            
            if label in sed1_events:
                new_list.append(li)
            
            elif label in sed2_events:
                pass
                
            else:
                raise Exception('Error!')
                
    with open(sed2_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
        
        for li in lis:
            label = li[3]
            
            if label in sed2_events:
                new_list.append(li)
                
            elif label in sed1_events:
                pass
                
            else:
                raise Exception('Error!')
               
               
    f = open(out_path, 'w')	# f = gzip.open('uuu.txt.gz', 'w')
    
    for li in new_list:
        f.write('\t'.join(li))
        f.write('\n')
        
    f.close()
    print('Write out to {}'.format(out_path))
                

if __name__ == '__main__':
    """Fuse results from sed1 (no SED) and sed2 (with SED). 
    Example: python utils/fuse_sed_results.py --workspace=$WORKSPACE --sed1_path=$WORKSPACE/submissions/main_pytorch/iteration=3000/submission_sed1.csv --sed2_path=$WORKSPACE/submissions/main_pytorch/iteration=3000/submission_sed2.csv
    """
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--sed1_path', type=str, required=True)
    parser.add_argument('--sed2_path', type=str, required=True)
    
    args = parser.parse_args()
    
    fuse_sed_results(args)