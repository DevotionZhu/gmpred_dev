from create_dataset import CreateDataset
from visualize_dataset import VisualizeDataset
from pre_process_dataset import CleanDataset

def main():
    #clean_ds = CleanDataset()
    #clean_ds.clean_dataset()
    #create_ds = CreateDataset()
    #create_ds.create_dataset()
    #create_ds.read_dataset()
    visualize_ds = VisualizeDataset()
    visualize_ds.visualize_features_labels_by_histogram()
    #visualize_ds.visualize_batch_dataset()
    #visualize_ds.visualized_1ts_dataset()
    #visualize_ds.visualized_seq_1ts_dataset()
    #visualize_ds.visualized_seq_dataset()
    #visualize_ds.visualize_all_dataset()


if __name__ == "__main__":
    main()
