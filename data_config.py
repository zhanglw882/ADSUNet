
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'Self':
            self.label_transform = "norm"
            self.root_dir = r'xxxxxxxxxx'
        elif data_name == 'inter_frame_data': ## 数据集的路径在这儿
            self.label_transform = "norm"
            self.root_dir = './IRSTD/inter_frame_data/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self

if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='inter_frame_data')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

