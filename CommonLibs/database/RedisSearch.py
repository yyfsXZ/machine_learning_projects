#!/usr/env/bin python
# -*- coding: utf-8 -*-
import logging
import hashlib
import rediscluster
import json
import sys
import urllib
import requests


class RedisSearch(object):
    def __init__(self):
        self._connection = None

    def initialize(self, nodes):
        """
        初始化connection
        :param nodes: [{"host": "172.0.0.1", "port": 6379}...]
        :return:
        """
        try:
            self._connection = rediscluster.StrictRedisCluster(startup_nodes=nodes, skip_full_coverage_check=True)
        except:
            logging.error("Failed to connect to redis cluster")
            sys.exit(0)
            return -1
        return 0

    def is_exist(self, key):
        """
        判断key是否在redis中
        :param key:
        :return:
        """
        try:
            return self._connection.exists(key)
        except:
            logging.error("rediscluster.StrictRedisCluster.exists method error")
            return False

    def is_exist_set(self, key, value):
        """
        检查value是否在key对应的集合中
        :param key:
        :param value:
        :return:
        """
        try:
            if not self.is_exist(key):
                logging.warning("No such key in redis")
                return True
            result = self._connection.sismember(key, value)
        except:
            logging.error("Failed to check member in SET")
            return False
        return result

    def is_exist_hashset(self, name, key):
        """
        检查key是否在名为name的hashset键中
        :param name:
        :param key:
        :return:
        """
        try:
            return self._connection.hexists(name, key)
        except:
            logging.error("rediscluster.StrictRedisCluster.hexists method error")
            return False

    def get_redis_string(self, key):
        """
        从string类型数据中获取value
        :param key:
        :return:
        """
        try:
            if not self.is_exist(key):
                logging.warning("No such key in redis")
                return -1, []
            result = self._connection.get(key)
        except:
            logging.error("Failed to get value from String type")
            return -2, []
        return 0, result

    def get_redis_set(self, key):
        """
        从集合类数据中获取全部value
        :param key:
        :return:
        """
        try:
            if not self.is_exist(key):
                logging.warning("No such key in redis")
                return -1, []
            result = list(self._connection.smembers(key))
        except:
            logging.error("Failed to get value from SET type")
            return -2, []
        return 0, result

    def get_redis_list(self, key):
        """
        从list类数据中获取全部value
        :param key:
        :return:
        """
        try:
            if not self.is_exist(key):
                logging.warning("No such key in redis")
                return -1, []
            result = self._connection.lrange(key, 0, -1)
        except:
            logging.error("Failed to get value from LIST type")
            return -2, []
        return 0, result

    def get_redis_hashset(self, name, key):
        """
        从hashset类数据中获取某一name下的key对应的值
        :param name:
        :param key:
        :return:
        """
        try:
            if not self.is_exist(name):
                logging.warning("Failed to get value from HASHSET type, name not exist")
                return -1, []
            elif not self.is_exist_hashset(name, key):
                logging.warning("Failed to get value from HASHSET type, key not exist")
                return -1, []
            result = self._connection.hget(name, key)
            result = json.loads(result)
        except:
            logging.error("Failed to get value from HASHSET type")
            return -2, []
        return 0, [elem.encode("utf-8") for elem in result]

    def get_redis_hashset_keys(self, name):
        """
        从hashset中获取所有key
        :param name:
        :return:
        """
        try:
            if not self.is_exist(name):
                logging.warning("Failed to get value from HASHSET type, name not exist")
                return -1, []
            result = self._connection.hgetall(name)
            return 0, result
        except:
            logging.warning("Failed to get HASHSET keys")
            return -2, []

    def write_redis_string(self, key, value):
        """
        写入string类型数据
        :param key:
        :param value:
        :return:
        """
        try:
            if self.is_exist(key):
                logging.warning("Exist a key named " + key)
            self._connection.set(key, value)
        except:
            logging.error("Failed to write STRING type data into redis")
            return -1
        return 0

    def write_redis_set(self, key, value):
        """
        写入Set类型数据
        :param key:
        :param value:
        :return:
        """
        try:
            self._connection.sadd(key, value)
        except:
            logging.error("Failed to write SET type data into redis")
            return -1
        return 0

    def write_redis_list(self, key, value):
        """
        写入Set类型数据
        :param key:
        :param value:
        :return:
        """
        try:
            self._connection.lpush(key, *value)
        except:
            logging.error(err)
            logging.error("Failed to write LIST type data into redis")
            return -1
        return 0

    def write_redis_list_hashset(self, name, key, value):
        """
        写入Hashset类型数据
        :param name:
        :param key:
        :param value:
        :return: 
        """
        try:
            value = json.dumps(value)
            self._connection.hset(name, key, value)
        except:
            logging.error("Failed to write HASHSET type data into redis")
            return -1
        return 0

    def md5sum(self, input):
        m2 = hashlib.md5()
        m2.update(input)
        return m2.hexdigest()

    def destory(self):
        #self._connection.save()
        self._connection = None
    
    def save(self):
        self._connection.save()

    def get_all_keys(self, pattern_="*"):
        return self._connection.keys(pattern=pattern_)

    def delete(self, key):
        #if "TermVector" not in key and "WhiteList" not in key:
        if "WhiteList" not in key:
            self._connection.delete(key)

    def get_list_len(self, key):
        return self._connection.llen(key)

    def delete_list_value(self, key, value, num=0):
        self._connection.lrem(key, value, num)

    def delete_set_value(self, key, value):
        try:
            if not self.is_exist(key):
                logging.warning("No such key in redis")
                return -1
            self._connection.srem(key, value)
        except:
            logging.error("Failed to get value from SET type")
            return -2
        return 0

    def reset_list_value(self, key, idx, value):
        self._connection.lset(key, idx, value)

    def remove_redis_set(self, key, value):
        self._connection.srem(key, value)

    def get_list_index_value(self, key, idx):
        return self._connection.lindex(key, idx)

    def delete_hashset_key(self, name, key):
        return self._connection.hdel(name, key)

    def query_hashset_keys(self, name):
        return self._connection.hkeys(name)


if __name__ == "__main__":
    # 单台
    #nodes = [{"host": "127.0.0.1", "port": 6379}]
    redisSearch = RedisSearch()
    redisSearch.initialize(nodes)
    
    sys.exit(0)






