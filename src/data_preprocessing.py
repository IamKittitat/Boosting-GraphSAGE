import os
import pandas as pd

def process_csv(file_path, output_path, class_output_path):
    df = pd.read_csv(file_path)
    
    df['disease'] = df['disease'].apply(lambda x: 0 if x == 'healthy' else 1)
    class_df = df[['disease']]
    df.drop(columns=['disease'], inplace=True)
    class_df.to_csv(class_output_path, index=False, header=False)

    cols_to_drop = ["subject_id", "class", "study"]
    df.drop(columns=cols_to_drop, inplace=True)
    
    df.to_csv(output_path, index=False, header=False)

def main():
    current_dir = os.path.dirname(__file__)
    input_file = os.path.join(current_dir, "../data/1_raw/GDMicro_T2D.csv")
    output_file = os.path.join(current_dir, "../data/2_OTU/GDMicro_T2D_data.csv")
    class_output_file = os.path.join(current_dir, "../data/2_OTU/GDMicro_T2D_class.csv")
    process_csv(input_file, output_file, class_output_file)

if __name__ == "__main__":
    main()