import pandas as pd
import wntr
import wntr.network.controls as controls
import shortuuid # Generate unique names for controls in WNTR


def copy_df(df):
    return df.copy()


def index_to_datetime(df, date_format):
    df.index = pd.to_datetime(df.index, format=date_format)
    return df


def remove_leading_trailing_spaces_column_names(df):
    df.columns = df.columns.str.replace(" ", "")
    return df


def apply_in_out_logic_zone(df, df_effect):
    return df.mul(df_effect, axis=1)


def drop_na_columnwise(df, thresh=0.7):
    thresh_abs = len(df)*thresh
    return df.dropna(axis=1, thresh=thresh_abs)


def sum_per_zone(df, df_zone):
    unique_zones = df_zone.unique().tolist()
    for zone in unique_zones:
        meters_in_zone = df_zone[df_zone==zone].index.tolist()
        
        df[zone] = df[meters_in_zone].sum(axis=1)
    
    df['all'] = df[unique_zones].sum(axis=1)
    return df


def rolling_per_zone(df, zone_names):
    for zone in zone_names:
        df[zone] = df[zone].rolling("15min").mean()
    
    return df


def interpolate_per_zone(df, zone_names):
    for zone in zone_names:
        df[zone] = df[zone].interpolate(method='time', limit_direction='both')
    
    return df


def pattern_per_zone(df, zones, periods_valve_open):
    for zone in zones:
        df[zone + '_final'] = df[zone].copy()
        for interval in periods_valve_open:
            # If valve is open, the pattern appplies to the entire zone
            df[zone + '_final'].loc[interval[0]:interval[1]] = df['all'].loc[interval[0]:interval[1]].copy() # L/s/prop
    return df


def get_mapping_active(master_df, mapping_type, 
                       modelname="epanet_name", 
                       inputname="input_name") -> dict:
    """
    Helper function to select subset from dataframe to
    - Simplify mapping internal Epanet names to the input names
    - Yield a standardized dict which can be used in following steps
    - Only select the tags which are set as Active
    
    Parameters
    ---------
    master_df: pd.DataFrame
        Pandas dataframe with at least 4 columns: 'type', 'active', modelname and inputname
    mapping_type: str
        String which is listed in 'type' column to differentiate between the different BCs/pattern mappings
    modelname: str
        String with the name which is internally used by the Epanet model of interest
    inputname: str
        String with the column name in the pandas.DataFrame which is used in subsequent
        functions to map to.
    
    Returns
    --------
    mapping_dict: dict
        Dict in which the keys are the internal Epanet names and the values are the external inputnames.
    """
    # Select subset of master data of certain type which are also active (meaning require updating)
    df_mapping = master_df[(master_df['type']==mapping_type) & (master_df['active']==True)]
    
    df_mapping = df_mapping.set_index(modelname)
    
    return df_mapping[inputname].to_dict()


def get_number_cp(wn, pattern_list):
    r"""
    Function to get the number of CPs which follow a certain list of patterns
    (OUTER JOIN). The number of CPs and mean_use_per_cp are important in a 
    next step to convert the zone-based use to a CP-based use.
    
    Parameters
    ---------
    wn: wntr.Model
        Configured wn model which requires update of CP useage data
    pattern_list: list
        List of strings containing the pattern names in which the CPs
        form one unified group
        
    Returns
    -------
    cp_num: int
        Total number of cps which follow the specified patterns
    mean_use_per_cp: float
        The average use per cp (same units as wn file)
    """
    # Initial placeholders prior to loop
    # Average use per cp (can be derived as minimum basevalue for all nodes in specific pattern)
    mean_use_per_cp = 1e9
    # Total use for all cps for specific pattern (prior to looping)
    tot_use = 0.
    # Loop over all nodes
    for node_name in wn.node_name_list:
        node_instance = wn.get_node(node_name)
        # Ignore nodes like reservoirs, ...
        if node_instance.node_type == "Junction":
            # Each Junction can have multiple timeseries
            # This is limited but occurs
            for node_demand in node_instance.demand_timeseries_list:
                # Only perform operations for selected patterns
                if node_demand.pattern_name in pattern_list:
                    # Add to total use
                    tot_use += node_demand.base_value
                    # The lowest non-zero use, is the use of 1 CP
                    mean_use_per_cp = min(mean_use_per_cp,
                                          node_demand.base_value)
    
    # Total number of CPs with a specific pattern_list
    cp_num = round(tot_use/mean_use_per_cp)
    
    return cp_num, mean_use_per_cp


def calculate_demand_patterns(wn, mapping_dict, pattern_df)->dict:
    r"""
    Based on an inputfile containing the zone-based patterns,
    the CP-based patterns can be calculated as the
    updated mean_use_per_cp.
    
    Parameters
    ---------
    wn: wntr.Model
        Configured wn model which requires update of CP useage data
    mapping_dict: dict
        Dict in which the keys are the internal Epanet names and the values are the external inputnames.
    pattern_df: pd.DataFrame
        DataFrame containing the demand patterns, which are 
        referenced as values in the mapping_dict and correctly sampled afo time.
        
    Returns
    ------
    output_dict: dict
        Dict which contains a dict per inputname. This subdict contains
        the original use per cp ('mean_use_per_cp_old'), the new use per cp
        ('mean_use_per_cp_new'), the ratio of new use of old ('mean_use_new_over_old')
        and the complete relative pattern as function of time ('relative_pattern') 
    
    """
    # Switch order of mapping dict to allow same pattern for different epanet_names
    # e.g. Blz_Noord and Blz_Zuid may be given same pattern
    pattern_dict = {value: [] for value in mapping_dict.values()}
        
    # Each input patternname get list of epanetnames which it needs to be mapped at
    for key,value in mapping_dict.items():
        pattern_dict[value].append(key)
        
    # 
    output_dict = {}
    for inputname, epanetname_list in pattern_dict.items():
        # First get number of cps which are part of this group in network file
        num_of_cp, mean_use_per_cp_old = get_number_cp(wn, epanetname_list)
        
        # 
        pattern_sel = pattern_df[inputname]
        mean_use_new = pattern_sel.mean()
        mean_use_per_cp_new = mean_use_new/num_of_cp
        
        #
        relative_pattern = pattern_sel/mean_use_new
        
        for epanetname in epanetname_list:
            output_dict[epanetname] = {'mean_use_per_cp_old': mean_use_per_cp_old,
                                       'mean_use_per_cp_new': mean_use_per_cp_new,
                                       'mean_use_new_over_old': mean_use_per_cp_new/mean_use_per_cp_old,
                                       'relative_pattern': relative_pattern}
        
    return output_dict


def assign_demand_patterns(wn, mapping_dict, pattern_df):
    r"""
    Assign the demand patterns to all relevant nodes.
    A relevant node is a node which has already been
    given a certain pattern, but the patterndata needs updating.
    
    Parameters
    ---------
    wn: wntr.Model
        Configured wn model which requires update of CP useage data
    mapping_dict: dict
        Dict in which the keys are the internal Epanet names and the
        values are the external inputnames.
    pattern_df: pd.DataFrame
        DataFrame containing the demand patterns, which are 
        referenced as values in the mapping_dict and correctly sampled afo time.
        
    Returns
    -------
    wn: wntr.Model
        Hydraulic model for which all demand patterns and base values have
        been updated
    """
    output_dict = calculate_demand_patterns(wn, mapping_dict, pattern_df)
    
    for node_name in wn.node_name_list:
        node_instance = wn.get_node(node_name)
        if node_instance.node_type == "Junction":
            for node_demand in node_instance.demand_timeseries_list:
                pattern_name = node_demand.pattern_name
                if (pattern_name!='') and (pattern_name in output_dict.keys()):
                    mean_use_new_over_old = output_dict[pattern_name]['mean_use_new_over_old']
                    node_demand.base_value *= mean_use_new_over_old
                
    
    for pattern_name,pattern_info in output_dict.items():
        wn.get_pattern(pattern_name).multipliers = pattern_info['relative_pattern'].values
    
    return wn


def assign_head_BC(wn, mapping_dict, pattern_df):
    """
    Assign the head patterns to all relevant nodes.
    A relevant node is a node which has already been
    given a certain pattern, but the patterndata needs updating.
    
    Parameters
    ---------
    wn: wntr.Model
        Configured wn model which requires update of head boundary conditions
    mapping_dict: dict
        Dict in which the keys are the internal Epanet names and the
        values are the external inputnames.
    pattern_df: pd.DataFrame
        DataFrame containing the head patterns, which are 
        referenced as values in the mapping_dict and correctly sampled afo time.
        
    Returns
    -------
    wn: wntr.Model
        Hydraulic model for which all head patterns and base heads have been updated
    """
    # Loop through all head names given in dict
    for key,value in mapping_dict.items():
        resv_instance = wn.get_node(key)
        # Assign new mean base value
        resv_instance.base_head = pattern_df[value].mean()

        # Assign new relative pattern 
        new_resv_pattern = (pattern_df[value]/pattern_df[value].mean()).values
        wn.get_pattern(resv_instance.head_pattern_name).multipliers = new_resv_pattern
        
    return wn


def assign_transfer_BC(wn, mapping_dict, factor_dict, pattern_df):
    r"""
    Assign the transfer patterns to all relevant nodes.
    A relevant node is a node which has already been
    given a certain pattern, but the patterndata needs updating.
    
    Parameters
    ---------
    wn: wntr.Model
        Configured wn model which requires update of transfer boundary conditions
    mapping_dict: dict
        Dict in which the keys are the internal Epanet names and the
        values are the external inputnames.
    factor_dict: dict
        Dict in which the keys are the internal Epanet names and the
        values are the factors (+1 or -1 to set the flow correctly).
    pattern_df: pd.DataFrame
        DataFrame containing the transfer patterns, which are 
        referenced as values in the mapping_dict and correctly sampled afo time.
        
    Returns
    -------
    wn: wntr.Model
        Hydraulic model for which all transfer patterns have been updated
    """
    # Flows in m3/H in input file => to be converted to m3/s for WNTR
    for modelname,inputname in mapping_dict.items():
        flow_pattern = pattern_df[inputname]/(3600)
        node_sel = wn.get_node(modelname)
        node_sel.demand_timeseries_list[0].base_value = factor_dict[modelname]
        wn.get_pattern(modelname).multipliers = flow_pattern
        
    return wn


def assign_valve_control(wn, control_mapping_dict, pattern_df):
    r"""
    Assign the correct binary control actions to all relevant valves.
    Valve controls are used here to avoid:
    - Flow leaving the netwerk at pressure boundaries (e.g. watertowers)
    
    This function only works for ON/OFF control. No semi-closed valves
    can be modelled using this function.
    
    Parameters
    ---------
    wn: wntr.Model
        Configured wn model which requires update of valve control settings
    mapping_dict: dict
        Dict in which the keys are the internal Epanet names and the
        values are the external inputnames.
    pattern_df: pd.DataFrame
        DataFrame containing the valve control patterns, which are 
        referenced as values in the mapping_dict and correctly sampled afo time.
        
    Returns
    -------
    wn: wntr.Model
        Hydraulic model for which all valve controls updated
    """
    for j,(name_WS,name_df) in enumerate(control_mapping_dict.items()):
        # Select pattern of interest
        df_sel = pattern_df[name_df].copy()
        # Convert datetime based index to time since start in seconds
        timedelta_index = (df_sel.index-df_sel.index[0]).total_seconds()
        df_sel.index = timedelta_index
        df_out = df_sel[df_sel.diff()[df_sel.diff()!=0.0].index]

        for i,(time,status) in enumerate(df_out.iteritems()):
            # Set up time condition when rule has to be applied
            cond1 = controls.SimTimeCondition(wn, '=', time)

            pipe = wn.get_link(name_WS)
            # Define which valve action needs to be applied (0=Closed, 1=Open)
            act1 = controls.ControlAction(pipe, 'status', status/100.)
            # Apply action for specified time => resulting in rule 
            unique_id = shortuuid.uuid()
            rule1 = controls.Rule(cond1, [act1], name='rule_{}'.format(unique_id))
            # Add rule to network
            wn.add_control('NewTimeControl_{}'.format(unique_id), rule1)

        link_ref = wn.get_link(name_WS)
        if df_sel.iloc[0] < 0.001:
            # Initial status which has to be written to file
            link_ref.initial_status = wntr.network.LinkStatus.Closed
            # Change status of current loaded state
            link_ref.status = wntr.network.LinkStatus.Closed
        else:
            # Initial status which has to be written to file
            link_ref.initial_status = wntr.network.LinkStatus.Opened
            # Change status of current loaded state
            link_ref.status = wntr.network.LinkStatus.Opened
    
    return wn