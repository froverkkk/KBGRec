import os
from configs.design import design_conf


class My_config:
    def __init__(self, config):
        self.__dict__ = config

    def save(self):
        with open(self.log_path, 'w', encoding='UTF-8') as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("{}, {}\n".format(key, str(value)))


def update_default_config(conf):
    config_func = conf.conf_name
    print(conf.conf_name)
    new_config = eval(config_func + '_conf')
    print(conf)
    for key, val in new_config.items():
        conf.__dict__[key] = val