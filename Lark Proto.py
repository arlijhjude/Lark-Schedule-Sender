# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 11:49:03 2025

@author: judecasenares.5
"""

import pandas as pd
import re
import numpy as np
from datetime import timedelta
import os
os.chdir(r'C:\Users\t1-adm-caseares.5\OneDrive - TP\Desktop\Lark Project')

def split_dataframe_by_date_header(df):
    """
    Splits a DataFrame into multiple DataFrames based on date headers 
    formatted as 'Day MM/DD' (e.g., 'Monday 12/8').

    The first column is assumed to be the Time/Identifier column and is 
    included in all resulting DataFrames.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary where keys are the date strings ('Monday 12/8', etc.)
              and values are the date-specific DataFrames.
    """
    # df = original_df
    date_dataframes = {} 
    if df.empty or len(df.columns) < 2:
        print("Warning: DataFrame is too small or empty.")
        return {}
        
    time_column = df.columns[0]
    
    # 🌟 NEW Regular Expression (The Fix)
    # 1. ^[A-Z]+: Starts with one or more uppercase letters (matching FRIDAY)
    # 2. [\s\xa0]+: Matches one or more of: regular space (\s) OR non-breaking space (\xa0)
    # 3. \d{1,2}/\d{1,2}$: Followed by MM/DD at the end of the string
    # We will still use .strip() on the column name for general robustness.
    # Note: Using [A-Za-z] would cover the FRIDAY, but the original image shows all caps, 
    # so matching all caps is safer. Let's make it case-agnostic for the day: [A-Za-z]
    date_pattern = r'^[A-Za-z]+\s*[\xa0]*\s*\d{1,2}/\d{1,2}$'

    # A simpler and highly robust approach is to strip and replace all whitespace with a single space:
    def normalize_header(header):
        # Convert to string, strip surrounding whitespace, then replace any internal
        # sequence of whitespace (including NBSP) with a single regular space.
        return re.sub(r'[\s\xa0]+', ' ', str(header).strip())
        
    # Pattern to match the normalized header (e.g., "FRIDAY 12/12")
    # This simplifies the regex needed to match the normalized string.
    normalized_date_pattern = r'^[A-Za-z]+\s\d{1,2}/\d{1,2}$'
    
    # Identify columns by normalizing the header before matching
    # We store the *original* column name for slicing later.
    date_columns = [col for col in df.columns if re.match(normalized_date_pattern, normalize_header(col))]

    if not date_columns:
        print("Warning: No 'DAY MM/DD' date headers found. Check column names.")
        return {'Original_Data': df}

    # Get the column index for each date header
    date_indices = [df.columns.get_loc(col) for col in date_columns]
    
    # Add the index of the column *after* the last column for the final slice end point
    date_and_end_indices = date_indices + [len(df.columns)]
    
    
    # Iterate through the date column blocks
    for i in range(len(date_columns)):
        # i = 0
        start_col_index = date_indices[i]
        end_col_index = date_and_end_indices[i+1]

        date_header = date_columns[i]
        
        # 1. Determine columns for the current slice
        columns_to_include = [time_column] + list(df.columns[start_col_index:end_col_index])

        # 2. Slice the DataFrame
        new_df = df.loc[:, columns_to_include].copy()
        
        # 3. Store the result
        date_dataframes[date_header] = new_df

    return date_dataframes, date_columns

def extract_schedule(df):
    """
    Extracts a long-format schedule from the wide-format DataFrame with 
    multi-level headers, identifying the agent, account, role, start time, 
    and end time for each scheduled block.
    """
    if df.empty or len(df.columns) < 2:
        return []
    
    
    # df = Compiled_Days
    
    
    
    # --- 1. CLEAN THE HEADER ROWS (Forward Fill) ---
    # The account (Row 0) and PM (Row 1) names are spread using merged cells.
    # We apply ffill horizontally (axis=1) to fill the NaNs.
    df.iloc[0:2] = df.iloc[0:2].fillna(method='ffill', axis=1, limit = 1)

    # --- 2. CREATE A SINGLE, DESCRIPTIVE HEADER ROW ---
    # Rows 0, 1, and 2 contain the Account, PM, and Role headers.
    
    # We use .astype(str) to handle potential mixed types and NaNs during combination.
    account_row = df.iloc[0].astype(str).str.strip()
    pm_row = df.iloc[1].astype(str).str.strip()
    role_row = df.iloc[2].astype(str).str.strip()
    
    new_columns = []
    
    # Iterate through all columns to create the new combined header name
    for i in range(len(df.columns)):
        account = account_row.iloc[i]
        pm = pm_row.iloc[i]
        role = role_row.iloc[i]
        
        # Combine into a single string: "Account | PM | Role"
        # We replace 'nan' strings (from empty cells) for clean output
        col_parts = [p for p in [account, pm, role] if p != 'nan']
        new_header = " | ".join(col_parts)
        
        # Ensure the Time column header is kept simple
        if i == 0:
            new_columns.append("Time")
        else:
            new_columns.append(new_header)
            
    # Apply the new headers to the DataFrame
    df.columns = new_columns
    
    # The actual data starts from Row 4 (index 3, as Row 3 contains 'Host', 'Operator' again)
    # The first image shows 'Time' starting at Excel Row 4 (index 3).
    data_df = df.iloc[3:].copy()
    data_df = data_df.reset_index(drop=True)
    
    # --- 3. MELT THE DATAFRAME TO LONG FORMAT ---
    # This turns the wide schedule data into a single column of agent/task names.
    schedule_long = data_df.melt(
        id_vars=['Time'],
        var_name='Header',
        value_name='Agent_Name'
    )
    
    # Drop rows where no agent is scheduled (i.e., the cell is NaN or empty)
    schedule_long = schedule_long.dropna(subset=['Agent_Name'])
    
    # --- 4. EXTRACT SCHEDULE DETAILS ---
    
    final_schedule = []
    
    # Convert 'Time' column to datetime objects for easy time calculations
    schedule_long['Start_Time'] = pd.to_datetime(schedule_long['Time'], format='%I:%M %p', errors='coerce').dt.time

    # Group by the unique schedule entries (Header and Agent_Name)
    # This groups all consecutive time slots for a single task/agent together
    for header, group in schedule_long.groupby('Header'):
        
        # Split the combined header back into its components
        header_parts = header.split(" | ")
        account = header_parts[0] if len(header_parts) > 0 else 'N/A'
        pm = header_parts[1] if len(header_parts) > 1 else 'N/A'
        role = header_parts[2] if len(header_parts) > 2 else 'N/A'
        
        # Group by the agent name
        for agent_name, agent_group in group.groupby('Agent_Name'):
            
            # Sort by time to ensure blocks are processed chronologically
            agent_group = agent_group.sort_values(by='Start_Time')
            
            # --- Identify Consecutive Time Blocks (The core logic) ---
            
            # Calculate the time difference between consecutive rows. 
            # We assume a 15-minute slot (7:00, 7:15, 7:30)
            agent_group['Time_Diff'] = agent_group['Start_Time'].apply(lambda x: pd.to_timedelta(str(x))) \
                                       .diff().dt.total_seconds() / 60
            
            # A block break occurs if the time difference is NOT 15 minutes, 
            # or if it's the very first row (where Time_Diff will be NaN)
            agent_group['Block_ID'] = (agent_group['Time_Diff'] != 15.0).cumsum()

            # Process each continuous time block
            for block_id, block in agent_group.groupby('Block_ID'):
                
                # The start time is the 'Time' value of the first row in the block
                start_time_str = block['Time'].iloc[0]
                
                # The end time is 15 minutes AFTER the 'Time' value of the LAST row in the block
                time_object = block['Time'].iloc[-1]
                time_string = str(time_object)
                last_time = pd.to_datetime(time_string, format='mixed')
                end_time_str = (last_time + pd.Timedelta(minutes=15)).strftime('%I:%M %p').lstrip('0')
                
                final_schedule.append({
                    'Account': account,
                    'PM': pm,
                    'Role': role,
                    'Agent': agent_name,
                    'Start_Time': start_time_str,
                    'End_Time': end_time_str
                })
                
    return final_schedule
# --- Example Usage ---
def generate_schedule(dataframe):
    """Aggregates the detailed time logs into continuous work blocks."""
    
    # 1. Group by the necessary categories (Day, Account, PM)
    # dataframe = Agent_a
    grouped = dataframe.groupby(['Date', 'Account', 'PM', 'Role', 'Agent'])
    
    final_schedule = {}
    for name, group in grouped:
        date, account, pm, role, agent = name
        
        # Sort by start time to ensure correct block detection
        group = group.sort_values('Start_Time_dt').reset_index(drop=True)
        
        blocks = []
        if not group.empty:
            current_start = group['Start_Time_dt'].iloc[0]
            current_end = group['End_Time_dt'].iloc[0]
            
            # Iterate through the rows to find continuous blocks
            for i in range(1, len(group)):
                next_start = group['Start_Time_dt'].iloc[i]
                next_end = group['End_Time_dt'].iloc[i]
                
                # Check if the next segment starts exactly where the current one ended
                # Allowing for a 1-minute buffer to handle minor time log discrepancies
                if next_start <= current_end + timedelta(minutes=1):
                    # It's continuous, extend the block's end time
                    current_end = next_end
                else:
                    # Gap detected, save the current block and start a new one
                    blocks.append({
                        'Start': current_start.strftime('%I:%M %p').lstrip('0'),
                        'End': current_end.strftime('%I:%M %p').lstrip('0'),
                        'Account': account,
                        'PM': pm
                    })
                    current_start = next_start
                    current_end = next_end

            # Save the last block
            blocks.append({
                'Start': current_start.strftime('%I:%M %p').lstrip('0'),
                'End': current_end.strftime('%I:%M %p').lstrip('0'),
                'Account': account,
                'PM': pm
            })
        
        # Store blocks under their date
        if date not in final_schedule:
            final_schedule[date] = []
        final_schedule[date].extend(blocks)
        
    return final_schedule

# --- 3. Format the Output Message ---
def create_schedule_message(schedule_data, team_member_name="Isabel"):
    """Formats the aggregated schedule data into the target message style."""
    # schedule_data = aggregated_schedule
    # Extract the date range (e.g., "12/9 - 12/11")
    dates = sorted(schedule_data.keys(), key=lambda x: pd.to_datetime(x.split()[-1], format='%m/%d'))
    start_date_str = dates[0].split()[-1]
    end_date_str = dates[-1].split()[-1]
    
    # 1. Header and Date Range
    message_parts = [
        f"**@{team_member_name} Cue the drum roll… your next week’s schedule has arrived!**",
        f"**Date Range: {start_date_str} - {end_date_str}**"
    ]
    
    # 2. Iterate and Format Blocks per Day
    for date_full in dates:
        # date_full = 'TUESDAY 12/9'
        day_of_week = date_full.split()[0].title()
        message_parts.append(f"\n**{day_of_week}**")
        
        # Sort blocks by start time for the current day
        day_blocks = sorted(schedule_data[date_full], key=lambda x: pd.to_datetime(x['Start'], format='%I:%M %p'))

        for block in day_blocks:
            # print(block)
            # We assume a standard 20-minute prep and 10-minute wrap as per the example message
            # You can customize these durations.
            PREP_MINUTES = 15
            WRAP_MINUTES = 10
            
            # Convert times to datetime objects for easy calculation
            start_time_dt = pd.to_datetime(block['Start'], format='%I:%M %p')
            end_time_dt = pd.to_datetime(block['End'], format='%I:%M %p')

            # Calculate Prep and Wrap times
            prep_start = start_time_dt.strftime('%I:%M %p').lstrip('0')
            prep_end = (start_time_dt + timedelta(minutes=PREP_MINUTES)).strftime('%I:%M %p').lstrip('0')
            wrap_start = (end_time_dt - timedelta(minutes=WRAP_MINUTES)).strftime('%I:%M %p').lstrip('0')
            wrap_end = end_time_dt.strftime('%I:%M %p').lstrip('0')

            account_name = block['Account']
            
            # Add Prep
            message_parts.append(f" {prep_start} – {prep_end} {account_name} LIVE Prep")
            
            # Add LIVE Block
            message_parts.append(f" **{prep_end} – {wrap_start} {account_name} LIVE**")

            # Add Wrap
            message_parts.append(f" {wrap_start} – {wrap_end} {account_name} LIVE Wrap")

    return "\n".join(message_parts)

# 1. Create a dummy DataFrame mimicking the new column names
# RawSchedule = pd.read_excel()
original_df = pd.read_excel(
        r"Test-Schedule Planner.xlsx",
        header=1)
# Rename the first column to match the user's 'Unnamed' context
original_df.columns = ['Unnamed: 0'] + list(original_df.columns[1:])
# 2. Run the splitting function
date_dataframes, Date_columns = split_dataframe_by_date_header(original_df)

Weeklydata = pd.DataFrame()
for Dates in Date_columns:
    temporaryData = date_dataframes[Dates]
    # 2. Run the extraction function
    schedule_data = extract_schedule(temporaryData)
    df_schedule = pd.DataFrame(schedule_data)
    has_numeric = df_schedule['Agent'].astype(str).str.contains('\d')
    df_schedule = df_schedule[~has_numeric].copy()
    df_schedule['Date'] = Dates
    
    Weeklydata = pd.concat([Weeklydata, df_schedule])

grouped_by_agent = Weeklydata.groupby('Agent')


Agent_list = Weeklydata['Agent'].unique() #Agent list
Agentname = Agent_list[0] #Virtual Agent as Parameter in checking and sending schedule.


Agent_a = grouped_by_agent.get_group(f'{Agentname}')

Agent_a['Start_Time_dt'] = pd.to_datetime(Agent_a['Start_Time'], format='%H:%M:%S')
Agent_a['End_Time_dt'] = pd.to_datetime(Agent_a['End_Time'], format='%I:%M %p')


# --- 4. Execution ---
aggregated_schedule = generate_schedule(Agent_a)
output_message = create_schedule_message(aggregated_schedule, f'{Agentname}')

print(output_message)











































