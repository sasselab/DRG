import numpy as np
import pandas as pd
import os
import argparse
import pickle
from memelite import tomtom
import tangermeme.io
from statsmodels.stats.multitest import multipletests

def map_from_expr_cell_types_to_lineage(expr_data,lineage_frame):
    """
    Get mapping from the cell types present in the expression data to their respective lineages to be able to group expression data by lineage. 
    -- expr_data is a pandas df with rows that are genes (ex, 'Zyg11a') and columns that are cell types (ex, 'B.B1a.PC.F#1').
    -- lineage_frame is a pandas df with columns 'cell_type' (ex, 'NKT.Sp.LPS.18hr') and 'lineage' (ex, 'abT'). 
    """
    shortened_lineage_ct=lineage_frame['cell_type'].str.lower()
    lineage_frame['shortened']=shortened_lineage_ct
    lineage_frame = lineage_frame.set_index("shortened")

    expr_data_lineage_mapping={}

    for item in expr_data.columns: 
        shortened_expr=".".join(item.split(".")[:-1]).lower()
        if shortened_expr in lineage_frame.index.values: 
            lineage_res=lineage_frame.loc[[shortened_expr],'lineage'].values[0]
            expr_data_lineage_mapping[item]=lineage_res
    return expr_data_lineage_mapping


def parse_seqlet_locations(input_string):
    """
    Parse file containing information about seqlet locations.
    """    
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


def get_per_lineage_tomtom_matches(expr_data_path, lineage_data_path, motif_database_path, seqlet_cwm_path, lineage_list, expression_threshold=.1,qval_threshold=0.05):
    """
    Get per-lineage tomtom matches by first filtering the provided motif database to only include genes with expression > expression_threshold in the expression data. 
    -- seqlet_cwm_path is the path to the .meme file containing query seqlet CWMs 
    -- lineage_list is the list of lineages for which to do this lineage-specific mapping for 
    """

    lineage_frame = pd.read_table(lineage_data_path, header = None, names = ['cell_type', 'lineage'])
    lineage_frame['lineage'] = lineage_frame['lineage'] .str.replace(' ', '_', regex=False) # Dendritic cell --> Dendritic_cell

    expr_data = pd.read_csv(expr_data_path,skiprows=2,sep='\t',index_col=0).iloc[:,1:] 
    expr_data=np.log(expr_data)

    mapping = map_from_expr_cell_types_to_lineage(expr_data,lineage_frame)
    mapping_series = pd.Series(mapping)
    mapping_series = mapping_series[mapping_series.index.isin(expr_data.columns)]
    grouped_expr = expr_data[mapping_series.index].groupby(mapping_series, axis=1).mean() # group expression data by using mapping from expression data cell types to lineages 

    database_motifs = tangermeme.io.read_meme(motif_database_path)  
    shortened_database_motif_names = [item.split('_')[2].lower() for item in database_motifs.keys()] # from, for ex, 'ENSMUSG00000079808_LINE1933_AC1689771_I' to 'ac1689771' (to match with the expression data gene names)

    expr_set = set(grouped_expr.index.str.lower()) # genes found in the expression data
    motifs_found_in_expr=[item for item in shortened_database_motif_names if item in expr_set] # database motifs found in the expression data 
    grouped_expr.index = grouped_expr.index.str.lower()
    final_filtered_grouped_expr=grouped_expr.loc[motifs_found_in_expr] # grouped expression data for genes present in motif database 

    query_motif_dict = tangermeme.io.read_meme(seqlet_cwm_path) 
    lineage_specific_tomtom_mapping_dict={} # build up dictionary containing lineage-specific dictionary mapping query --> target motifs  

    for lineage in lineage_list: 
        print(f'getting {lineage} tomtom mapping')
        if lineage in final_filtered_grouped_expr.columns: 
            above_threshold_database_motif_names = set(final_filtered_grouped_expr[final_filtered_grouped_expr[lineage]>expression_threshold].index.values)
            lineage_specific_target_dict = {k: v for k, v in database_motifs.items() if k.lower().split('_')[2] in above_threshold_database_motif_names} # only use these in tomtom
        else: 
            print(f'{lineage} not found in expression data, using all database motifs in tomtom matching')
            lineage_specific_target_dict=database_motifs # match to all database motifs 
        lineage_specific_tomtom_mapping = get_tomtom_matches(lineage_specific_target_dict, query_motif_dict,qval_threshold=qval_threshold)
        lineage_specific_tomtom_mapping_dict[lineage] = lineage_specific_tomtom_mapping

    return lineage_specific_tomtom_mapping_dict


def get_tomtom_matches(target_motif_dict, query_motif_dict,qval_threshold=0.05):
    """
    Use tomtom to match query motifs to target motifs. Return a dictionary for each query that contains a target match with < qval_threshold. The dictionary maps 
    each query to a list of all target matches below qval_threshold (sepearted by ',').
    """

    target_names = np.array(list(target_motif_dict.keys()))
    target_values = list(target_motif_dict.values())

    query_names = np.array(list(query_motif_dict.keys()))
    query_values = list(query_motif_dict.values())
    
    print(f'searching {len(query_names)} queries against {len(target_names)} targets')
    p, scores, offsets, overlaps, strands = tomtom(query_values, target_values)

    # convert p value to q values 
    _, q, _, _ = multipletests(p.flatten(), method='fdr_bh')
    q = q.reshape(p.shape[0], p.shape[1])
    
    # find query indices where there is a target match below threshold 
    below_threshold_query_idxs = np.where(np.min(q,axis=1)<qval_threshold)[0]

    # get query names associated with below threshold idxs 
    below_threshold_query_names = np.char.strip(query_names[below_threshold_query_idxs]) # strip bc query names are of the form '0 ' (trailing space)

    # get target names associated with below theshold matches 
    target_match_names = []
    for query_idx in below_threshold_query_idxs:
        curr_qvals = q[query_idx, :]  
        below_threshold_idxs = np.where(curr_qvals < qval_threshold)[0] 
        sorted_idxs = below_threshold_idxs[np.argsort(curr_qvals[below_threshold_idxs])]
        target_names_below_threshold = [target_names[idx].split('_')[2] if '_' in target_names[idx] else target_names[idx] for idx in sorted_idxs]
        target_match_string = ",".join(target_names_below_threshold)
        target_match_names.append(target_match_string)  
    target_match_names=np.array(target_match_names)

    query_to_target_dict = dict(zip(below_threshold_query_names, target_match_names))

    return query_to_target_dict


def save_bed_of_motif_matches(save_dir, seqlet_info_path, seqlet_cwm_path, pos_info_path, expr_data_path, lineage_data_path, motif_database_path, label_top_match_only=False,lineage_specific_tomtom=True,expression_threshold=.1,qval_threshold=0.05):
    """
    Save .bed files labeling seqlet locations with motif matches from tomtom below qval_threshold. 
    The number of .bed files saved will be number of sequences x number of tracks. 
    -- save_dir is the directory in which the .bed files (and the tomtom mapping dictionary) will be saved
    -- seqlet_cwm_path is the path to the .meme file containing query seqlet CWMs 
    -- pos_info_path is the path to the pickle file containing position info for the current sequence(s), ex: {'Cd19': array(['chr7', '126414800', '126416800']}
    -- seqlet_info_path is the path to the file containing location information for each seqlet 
    -- if label_top_match_only is True, the .bed will only contain the top (i.e. lowest q-value) match: i.e. 
        chr7	126416250	126416259	Rorc
        instead of 
        chr7	126416250	126416259	Rorc,Nr2c1,Esrra,Rarb,Esrrg,Esrrb,Rorb,Rxrb,Rxrg,Nr1d2,Esr2,Nr2f2,Nr2f6,Nr5a2,Nr1h5,Rara
    -- if lineage_specific_tomtom is True, the tomtom matching will be done in a lineage-specific way -- i.e. by first filtering out genes from the motif database with expression < expression_threshold 
    """
    
    os.makedirs(save_dir, exist_ok=True)

    seqlet_info = pd.read_csv(seqlet_info_path,header=None)
    seqlet_info_split = pd.DataFrame(
        [parse_seqlet_locations(loc) for loc in seqlet_info[0].values],
        columns=["seq_names", "track_info", "motif_starts", "motif_ends", "motif_ids"]
    )

    lineages = list(set(['.'.join(item.split('.')[:-1]) for item in seqlet_info_split['track_info']])) # ex, B.ATAC --> B

    if lineage_specific_tomtom: 
        print(f'getting lineage-specific tomtom matches for lineages:{lineages}')
        tomtom_dict = get_per_lineage_tomtom_matches(expr_data_path=expr_data_path, lineage_data_path=lineage_data_path, motif_database_path=motif_database_path, lineage_list=lineages, expression_threshold=expression_threshold,qval_threshold=qval_threshold,seqlet_cwm_path=seqlet_cwm_path)
    else: 
        print('getting non-lineage-specific tomtom matches')
        target_motif_dict = tangermeme.io.read_meme(motif_database_path)  
        query_motif_dict = tangermeme.io.read_meme(seqlet_cwm_path)
        tomtom_dict = get_tomtom_matches(target_motif_dict, query_motif_dict,qval_threshold=qval_threshold)

    query_to_target_dict_path=f"{save_dir}tomtom_match_dict.pkl"
    print(f'saving tomtom match dictionary to {query_to_target_dict_path}')
    with open(query_to_target_dict_path, "wb") as f:
        pickle.dump(tomtom_dict, f)

    with open(pos_info_path, 'rb') as f:
        pos_info = pickle.load(f)
    
    if label_top_match_only:
        if lineage_specific_tomtom:
            # update each lineage dict
            tomtom_dict = {
                lineage: {q: v.split(',')[0] for q, v in lineage_dict.items()}
                for lineage, lineage_dict in tomtom_dict.items()
            }
        else:
            # update top-level dict
            tomtom_dict = {q: v.split(',')[0] for q, v in tomtom_dict.items()}
        
    print(f'saving .bed files to {save_dir}')
    for seq_id in list(set(seqlet_info_split['seq_names'])): 
        seq_rows = seqlet_info_split[seqlet_info_split['seq_names']==seq_id]
        seq_start_pos = int(pos_info[seq_id][1])
        for track_id in list(set(seq_rows['track_info'])):
            track_rows = seq_rows[seq_rows['track_info']==track_id]
            curr_chr = np.array([pos_info[seq_id][0]] * len(track_rows))
            motif_starts = track_rows['motif_starts'].values + seq_start_pos
            motif_ends = track_rows['motif_ends'].values + seq_start_pos
            lineage='.'.join(track_id.split('.')[:-1])
            if lineage_specific_tomtom: 
                curr_tomtom_dict=tomtom_dict[lineage]
            else:
                curr_tomtom_dict=tomtom_dict
            bed_rows = [
                [
                    curr_chr[i],
                    motif_starts[i],
                    motif_ends[i],
                    curr_tomtom_dict[track_rows['motif_ids'].values[i]]
                    if track_rows['motif_ids'].values[i] in curr_tomtom_dict
                    else f"cluster {track_rows['motif_ids'].values[i]}" # label with cluster {cluster_idx} if no significant match found 
                ]
                for i in range(len(track_rows))]
            
            bed_file_path = f'{save_dir}{seq_id}_{track_id}.bed'
            with open(bed_file_path, "w") as bed_file:
                for row in bed_rows:
                    bed_file.write("\t".join(map(str, row)) + "\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--motif_database_path', type=str, required=True)
    parser.add_argument('--seqlet_cwm_path', type=str, required=True)
    parser.add_argument('--seqlet_info_path', type=str, required=True)
    parser.add_argument('--lineage_data_path', default='')
    parser.add_argument('--expr_data_path', default='')
    
    parser.add_argument('--expression_threshold', type=float, default=0.1)
    parser.add_argument('--lineage_specific_tomtom', type=int, default=1)
    parser.add_argument('--qval_threshold', type=float, default=0.05)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--pos_info_path', type=str, required=True)
    parser.add_argument('--label_top_match_only', type=int, default=0)

    args = parser.parse_args()

    save_bed_of_motif_matches(save_dir=args.save_dir, seqlet_info_path=args.seqlet_info_path, seqlet_cwm_path=args.seqlet_cwm_path, pos_info_path=args.pos_info_path, expr_data_path=args.expr_data_path, lineage_data_path=args.lineage_data_path, motif_database_path=args.motif_database_path, label_top_match_only=args.label_top_match_only,lineage_specific_tomtom=args.lineage_specific_tomtom,expression_threshold=args.expression_threshold,qval_threshold=args.qval_threshold)


    
