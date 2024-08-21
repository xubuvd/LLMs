# -*- coding:utf-8 -*-
# @Time       :2023/1/10 10:09
# @AUTHOR     :YUNYI
# @SOFTWARE   :xy_py_project
# @DESC       : doris操作帮助
import base64
import json
import logging
import time
from pymysql.converters import escape_string
import pymysql
from pymysql.cursors import DictCursorMixin
from dbutils.pooled_db import PooledDB


def Logger(name=__name__, filename=None, level='INFO', filemode='a'):
    """
    定义日志
    :param name:
    :param filename: filename string
    :param level:
    :param filemode:
    :return:
    """
    logging.basicConfig(
        filename=filename,
        filemode=filemode,
        level=level,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    if filename:
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


DorisLogger = Logger(name='DorisClient')


def retry(*args, **kwargs):
    """
    定义重试机制
    :param args:
    :param kwargs:
    :return:
    """
    max_retry = kwargs.get('max_retry', 3)
    retry_diff_seconds = kwargs.get('retry_diff_seconds', 3)

    def warpp(func):
        def run(*args, **kwargs):
            for i in range(max_retry + 1):
                if i > 0:
                    DorisLogger.warning(f"will retry after {retry_diff_seconds} seconds，retry times : {i}/{max_retry}")
                time.sleep(retry_diff_seconds)
                flag = func(*args, **kwargs)
                if flag:
                    return flag

        return run

    return warpp


class DorisSession(object):
    def __init__(self, doris_config):
        """
        doris链接信息初始化
        :param doris_config: doris链接初始化
        doris_config = {'fe_servers': [],
                        'database': xxxx,
                        'user': xxxx,
                        'passwd': xxxx,
                        'prot':9030,
                        'charset':'utf-8'
                        }
        """

        fe_servers = doris_config.get('fe_servers')
        database = doris_config.get('database')
        user = doris_config.get('user')
        passwd = doris_config.get('passwd')
        prot = doris_config.get('prot')
        charset = doris_config.get('charset')
        assert fe_servers
        assert database
        assert user
        assert passwd
        self.fe_servers = fe_servers
        self.database = database
        self.user = user
        self.passwd = passwd
        self.port = prot
        self.charset = charset
        self.Authorization = base64.b64encode((user + ':' + passwd).encode('utf-8')).decode('utf-8')
        # self.doris_cfg = {
        #     'host': fe_servers[0].split(':')[0],
        #     'port': prot,
        #     'database': database,
        #     'passwd': passwd
        # }
        self.doris_pool = PooledDB(
            creator=pymysql,
            maxconnections=20,
            mincached=2,
            maxcached=5,
            maxshared=1,
            blocking=True,
            maxusage=None,
            setsession=[],
            ping=0,
            host=self.fe_servers[0].split(':')[0],
            port=self.port,
            user=self.user,
            password=self.passwd,
            database=self.database,
            charset=self.charset
        )
        self.conn = None
        self.cursor = None

    def _connect(self):
        """
        创建数据库连接
        :return:
        """
        try:
            self.conn = self.doris_pool.connection()
            self.cursor = self.conn.cursor(pymysql.cursors.DictCursor)
            return True
        except Exception as e:
            DorisLogger.error("Doris数据库连接异常......")
            return False

    def _close_conn(self):
        """
        关闭连接
        :return:
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def execute_sql(self, sql, *args):
        """
        sql语句执行 插入/更新/删除等
        :param sql:
        :param args:
        :return:
        """
        try:
            self._connect()
            DorisLogger.info("msg: %s" % sql)
            self.cursor.execute(sql, args)
            self.conn.commit()
        except Exception as e:
            DorisLogger.error('msg: %s' % e)
        finally:
            self._close_conn()

    def insert_many(self, sql, data):
        """
        批量插入数据
        """
        try:
            self._connect()
            print(sql)
            self.cursor.executemany(sql, data)
            self.conn.commit()
        except Exception as e:
            DorisLogger.error("msg: %s " % e)
        finally:
            self._close_conn()

    def insert_one(self, sql, data):
        """
        批量插入数据
        """
        try:
            self._connect()
            DorisLogger.info("msg: %s " % sql)
            self.cursor.execute(sql, data)
            self.conn.commit()
        except Exception as e:
            DorisLogger.error('msg: %s ' % e)
        finally:
            self._close_conn()

    def select(self, sql, *args):
        """
        执行查询语句
        """
        try:
            self._connect()
            DorisLogger.info("msg: % s" % sql)
            self.cursor.execute(sql, args)
            res = self.cursor.fetchall()
            return res
        except Exception as e:
            DorisLogger.error("msg: %s " % e)
            return False
        finally:
            self._close_conn()


if __name__ == '__main__':
    doris_config = {'fe_servers': ['172.16.1.93:19030'],
                    'database': 'test',
                    'user': 'xxxxx',
                    'passwd': 'xxxxxx',
                    'prot': 19030,
                    'charset': 'utf8'
                    }
    doris_client = DorisSession(doris_config=doris_config)
    # 执行查询
    DorisLogger.info("执行查询")
    res = doris_client.select(sql="select * from test.example_tbl")
    print(res)

    # # 执行建表语句
    # create_sql = """
    # CREATE TABLE IF NOT EXISTS test.example_tbl_user_agg
    # (
    #     `user_id` LARGEINT NOT NULL COMMENT '用户id',
    #     `username` VARCHAR(50) NOT NULL COMMENT '用户昵称',
    #     `city` VARCHAR(20) REPLACE COMMENT '用户所在城市',
    #     `age` SMALLINT REPLACE COMMENT '用户年龄',
    #     `sex` TINYINT REPLACE COMMENT '用户性别',
    #     `phone` LARGEINT REPLACE COMMENT '用户电话',
    #     `address` VARCHAR(500) REPLACE COMMENT '用户地址',
    #     `register_time` DATETIME REPLACE COMMENT '用户注册时间'
    # )
    # AGGREGATE KEY(`user_id`, `username`)
    # DISTRIBUTED BY HASH(`user_id`) BUCKETS 1
    # PROPERTIES (
    # 'replication_allocation' = 'tag.location.default: 1'
    # )
    # """
    #
    # DorisLogger.info("执行建表语句")
    # # doris_client.execute_sql(sql=create_sql)
    #
    # DorisLogger.info("执行插入单条数据")
    # data = [
    #     {"user_id": 10020,"nickname": "张三","json_data": {"cets":111}},
    #     {"user_id": 10032,"username": "张三","json_data": {"cets":111}}
    # ]
    # aa = {"json_data": {"cets":111}}
    # d = json.dumps(aa)
    # print(d)
    #
    # # aa = "insert into test.ods_test_test (user_id,nickname,json_data) values ({user_id},{nickname},{json_data})"
    # # print(aa.format(json=escape_string(d,d)))
    # # doris_client.insert_one(
    # #     sql='insert into test.ods_test_test (user_id,nickname,json_data) values (%(user_id)s,%(nickname)s,%(json_data)s)',
    # #     data=data[0])
    #
    # print("insert into test.ods_test_test (user_id,nickname,json_data) values (%(user_id)s,%(nickname)s,%(json_data)s)" % (
    #     data[0]))
