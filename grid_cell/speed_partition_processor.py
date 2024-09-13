
class Speed_Partition_Processor():
    def __init__(self, feature_map, label, speed_label):
        '''
        input:
            feature_map: (n_samples, n_features)
            label: (n_samples, n_labels)
            speed_label: (n_samples)
        '''
        self.feature_map = feature_map
        self.label = label
        self.speed_label = speed_label
        self.partitions = []

    def load_speed_win(self, speed_win):
        '''
        input:
            speed_win: list of tuples, each tuple is a speed window
        '''
        self.partitions = []  # Clear previous partitions
        for win in speed_win:
            # Create a boolean mask where the speed_label falls within the current speed window
            mask = (self.speed_label >= win[0]) & (self.speed_label < win[1])
            # Apply the mask to partition feature_map and label
            self.partitions.append((self.feature_map[mask], self.label[mask]))

    def apply_on_speed_win(self, func, *args, **kwargs):
        '''
        Apply func to each partition
        input:
            func: function to apply to each partition. func must have feamap and label as input, or feamap along as a input
        output:
            results: list of results from applying func to each partition
        '''
        results = []
        for feamap, label in self.partitions:
            try:
                results.append(func(feamap=feamap, label=label, *args, **kwargs))
            except:
                results.append(func(feamap=feamap, *args, **kwargs))
        return results
