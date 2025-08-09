import numpy as np
import pandas as pd
import os
import argparse
import pickle
from tangermeme.tools.tomtom import tomtom
from tangermeme.io import read_meme
from statsmodels.stats.multitest import multipletests

def parse_motif_locations(input_string):
    # parse file containing info on motif locations 
    
    seq_name = input_string.split("_")[0]
    motif_id = input_string.split(" ")[-1]
    range_vals = input_string.split("_")[-1].split(' ')[0]
    start_range = int(range_vals.split('-')[0])
    end_range = int(range_vals.split('-')[1])
    track_info = input_string.split('_')[1:-1]
    if len(track_info)>1: 
        track_info=f'{track_info[0]}_{track_info[1]}'
    else: 
        track_info=track_info[0]
    return seq_name, track_info, start_range, end_range, motif_id


def get_tomtom_matches(motif_database_path, input_seqlet_path,qval_threshold=0.05):
    # Given motif_database_path (.meme) and input_seqlet_path (.meme), return an array of query names
    #   and an array of the target names for matches below qval_threshold. 
    # The input input_seqlet_path should be a .meme file after created after seqlets have already been extracted, 
    #   clustered, and normalized for use with tomtom -- see https://github.com/LXsasse/DRG/blob/main/examples/Attribution_analysis.md for this process. 
    # The array of target names returned includes 
    #   all target names below qval_threshold (there can be multiple for each query). The array of query names and 
    #   target names will be of the same length: for ex, below_threshold_query_names[0] and target_match_names[0]
    #   represents a match. 

    targets = read_meme(motif_database_path)
    target_names = np.array([name for name in targets.keys()])
    target_pwms = [pwm for pwm in targets.values()]
    
    queries = read_meme(input_seqlet_path)
    query_names = np.array([name for name in queries.keys()])
    query_pwms = [pwm for pwm in queries.values()]
    
    print(f'searching {len(queries)} queries against {len(targets)} targets')
    p, scores, offsets, overlaps, strands = tomtom(query_pwms, target_pwms)

    # convert p value to q values 
    #q = (p*(p.shape[0]*p.shape[1])).numpy()
    _, q, _, _ = multipletests(p.flatten(), method='fdr_bh')
    q = q.reshape(p.shape[0], p.shape[1])
    
    # find query indices where there is a target match below threshold 
    below_threshold_query_idxs = np.where(np.min(q,axis=1)<qval_threshold)[0]

    # get query names associated with below threshold idxs 
    below_threshold_query_names = np.char.strip(query_names[below_threshold_query_idxs])

    # get target names associated with below theshold matches 
    target_match_names = []
    for query_idx in below_threshold_query_idxs:
        curr_qvals = q[query_idx, :]  
        below_threshold_idxs = np.where(curr_qvals < qval_threshold)[0] 
        sorted_idxs = below_threshold_idxs[np.argsort(curr_qvals[below_threshold_idxs])]
        target_names_below_threshold = [target_names[idx].split('_')[2] for idx in sorted_idxs]
        target_match_string = ",".join(target_names_below_threshold)
        target_match_names.append(target_match_string)  
    target_match_names=np.array(target_match_names)
        
    return below_threshold_query_names,target_match_names
    

def save_bed_of_motif_matches(query_names, target_names, motif_locations_info_path,save_dir,pos_info_path, label_top_match_only=True,only_save_matches=False):
    # Save .bed files labeling seqlet locations with motif matches from tomtom. The number of .bed files 
    #   saved will be number of sequences x number of tracks. 
    # query_names and target_names are outputs from the function get_tomtom_matches. 
    # motif_locations_info_path is the file ending in 'corpva.txt' saved in the clustering seqlet step: see https://github.com/LXsasse/DRG/blob/main/examples/Attribution_analysis.md for details. 
    # save_dir is the directory in which to save the .bed files 
    # pos_info_path is a pickle file containing chromosome, start position, and end position for each sequence. 
    #   this file should have been created when the sequence npz was created in its same directory, by using 
    #   the argument --'save_pos_info' with the file generate_fasta_from_bedgtf_and_genome.py 
    # If label_top_match_only is True, only the lowest qval match will be saved as the label. If it is False, 
    #   the label will include all matches below the qval threshold. 
    # If only_save_matches is True, the .bed file only contains motifs whose clusters match the database below the threshold. IF False, all motifs are saved to the .bed. 
    
    os.makedirs(save_dir, exist_ok=True)
    
    with open(pos_info_path, 'rb') as f:
        pos_info = pickle.load(f)
    
    motif_locations = pd.read_csv(motif_locations_info_path,header=None)
    motif_locations_split = pd.DataFrame(
        [parse_motif_locations(loc) for loc in motif_locations[0].values],
        columns=["seq_names", "track_info", "motif_starts", "motif_ends", "motif_ids"]
    )
    
    query_to_target = {}
    for i in range(len(query_names)):         
        if label_top_match_only:
            query_to_target[query_names[i]] = target_names[i].split(',')[0] 
        else:
            query_to_target[query_names[i]] = target_names[i]

    if only_save_matches: 
        # select rows (seqlet locations) where there exists a tomtom match below threshold 
        matched_rows = motif_locations_split[motif_locations_split['motif_ids'].isin(query_names.astype(str))]
    else: 
        matched_rows=motif_locations_split
        
    # only save .bed files for seqs and tracks that have at least one match below threshold 
    for seq_id in list(set(matched_rows['seq_names'])): 
        seq_rows = matched_rows[matched_rows['seq_names']==seq_id]
        seq_start_pos = int(pos_info[seq_id][1])
        for track_id in list(set(seq_rows['track_info'])):
            track_rows = seq_rows[seq_rows['track_info']==track_id]
            curr_chr = np.array([pos_info[seq_id][0]] * len(track_rows))
            motif_starts = track_rows['motif_starts'].values + seq_start_pos
            motif_ends = track_rows['motif_ends'].values + seq_start_pos
            #bed_rows = [[curr_chr[i], motif_starts[i], motif_ends[i], query_to_target[track_rows['motif_ids'].values[i]]] for i in range(len(track_rows))]
            
            bed_rows = [
                [
                    curr_chr[i],
                    motif_starts[i],
                    motif_ends[i],
                    query_to_target[track_rows['motif_ids'].values[i]]
                    if track_rows['motif_ids'].values[i] in query_to_target
                    else track_rows['motif_ids'].values[i]
                ]
                for i in range(len(track_rows))]
            
            bed_file_path = f'{save_dir}{seq_id}_{track_id}.bed'
            with open(bed_file_path, "w") as bed_file:
                for row in bed_rows:
                    bed_file.write("\t".join(map(str, row)) + "\n")

                    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--motif_database_path', type=str, required=True)
    parser.add_argument('--input_seqlet_path', type=str, required=True)
    parser.add_argument('--qval_threshold', type=float, default=0.05)
    parser.add_argument('--motif_locations_info_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--pos_info_path', type=str, required=True)
    parser.add_argument('--label_top_match_only', type=int, default=1)
    parser.add_argument('--only_save_matches', type=int, default=0)

    args = parser.parse_args()
    
    print('getting tomtom matches')
    below_threshold_query_names,target_match_names = get_tomtom_matches(args.motif_database_path, args.input_seqlet_path,qval_threshold=args.qval_threshold)
    
    print('saving .bed files')
    save_bed_of_motif_matches(below_threshold_query_names, target_match_names, args.motif_locations_info_path,args.save_dir,args.pos_info_path, label_top_match_only=args.label_top_match_only,only_save_matches=args.only_save_matches)

    
    
