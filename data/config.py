from configparser import ConfigParser
import os
import re


class Configurable:

    def __init__(self, path, extra_args=None) -> None:
        """读取config文件中的配置 并将额外的参数添加到类属性中

        Args:
            path (str): config文件路径
            extra_args (list, optional): 额外的参数. Defaults to None.
        """
        self.config = ConfigParser()
        self.config.read(path, encoding="utf-8")

        if extra_args:
            self.add_extra_args(extra_args)

        # 加载文件中的参数
        sections = self.config.sections()
        for section in sections:
            items = self.config.items(section)
            self._add_attr(items)

        save_dir = self.config.get("Save", "save_dir")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.config.write(open(self.config.config_file, "w"))

    def add_extra_args(self, extra_args):
        """添加额外的参数"""
        extra_args = dict([
            (k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])
        ])
        for section in self.config.sections():
            for k, v in self.config.items(section):
                if k in extra_args:
                    v = extra_args.pop(k)
                    self.config.set(section, k, v)


    def _add_attr(self, items):
        """将参数添加到类属性中

        Args:
            items (dict): k: 属性名 v: 属性值
        """
        num_value = re.compile(r"^[-+]?[0-9]+\.[0-9]+$")
        for k, v in items:
            try:
                v = int(v)
            except:
                try:
                    v = float(v)
                except:
                    if "None" in v:
                        v = None
                    elif "True" == v or "False" == v:
                        v = True if "True" == v else False

            self.__setattr__(k, v)
