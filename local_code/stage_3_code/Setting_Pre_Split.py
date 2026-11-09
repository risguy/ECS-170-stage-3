'''
Concrete SettingModule class for datasets that already ship with a train/test partition
'''

from local_code.base_class.setting import setting


class Setting_Pre_Split(setting):

    def load_run_save_evaluate(self):
        # load dataset, already partitioned into {'train': {...}, 'test': {...}}
        loaded_data = self.dataset.load()

        # run MethodModule
        self.method.data = loaded_data
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None
