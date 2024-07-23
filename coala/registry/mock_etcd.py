import threading
try:
    import queue
except ImportError:
    import Queue as queue


class MockEtcd(object):
    class KV(object):
        def __init__(self, key, value):
            self._key = key
            self._value = value

        @property
        def key(self):
            if isinstance(self._key, str):
                return self._key.encode()
            return self._key

        @property
        def value(self):
            if isinstance(self._value, str):
                return self._value.encode()
            return self._value

    class EventNotifier(object):
        def __init__(self, clnt):
            self._queue = queue.Queue()
            self._clnt = clnt

        def get_client_belongto(self):
            return self._clnt

        def notify(self, key, value):
            self._queue.put(MockEtcd.KV(key, value))

        def wait_for_event(self):
            while True:
                event = self._queue.get()
                if event is None:
                    return
                yield event

        def cancel(self):
            self._queue.put(None)

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}
        self._event_notifier = {}

    def get(self, key):
        with self._lock:
            if key in self._data:
                if isinstance(self._data[key], str):
                    return self._data[key].encode(), None
                return self._data[key], None
            return None, None

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
            self._notify_if_need(key)

    def delete(self, key):
        with self._lock:
            self._data.pop(key, None)
            self._notify_if_need(key)

    def delete_prefix(self, prefix):
        with self._lock:
            deleted = []
            for key in self._data:
                if key.startswith(prefix):
                    deleted.append(key)
            for key in deleted:
                self._data.pop(key, None)
                self._notify_if_need(key)

    def put_if_not_exists(self, key, value):
        with self._lock:
            if key in self._data:
                return False
            self._data[key] = value
            self._notify_if_need(key)
            return True

    def replace(self, key, old_value, new_value):
        with self._lock:
            stored = None
            if key in self._data:
                stored = self._data[key]
            if stored != old_value:
                return False
            self._data[key] = new_value
            self._notify_if_need(key)
            return True

    def watch(self, key, clnt):
        with self._lock:
            en = MockEtcd.EventNotifier(clnt)
            if key not in self._event_notifier:
                self._event_notifier[key] = [en]
            else:
                self._event_notifier[key].append(en)
            return en.wait_for_event(), en.cancel

    def close(self, clnt):
        with self._lock:
            for key in self._event_notifier:
                self._event_notifier[key] = [
                        en for en in self._event_notifier[key] if
                        en.get_client_belongto() == clnt
                    ]

    def get_prefix(self, prefix, sort_order='ascend'):
        kvs = []
        with self._lock:
            for key, value in self._data.items():
                if key.startswith(prefix):
                    kvs.append((value.encode(), MockEtcd.KV(key, None)))
            if sort_order == 'descend':
                kvs = sorted(kvs, key=lambda kv: kv[1].key, reverse=True)
            elif sort_order == 'ascend':
                kvs = sorted(kvs, key=lambda kv: kv[1].key, reverse=False)
            return kvs

    def _notify_if_need(self, key):
        if key in self._event_notifier:
            value = None
            if key in self._data:
                value = self._data[key]
            for en in self._event_notifier[key]:
                en.notify(key, value)


class MockEtcdClient(object):
    POOL_LOCK = threading.Lock()
    MOCK_ETCD_POOL = {}

    def __init__(self, host, port):
        key = '{}:{}'.format(host, port)
        with self.POOL_LOCK:
            if key not in self.MOCK_ETCD_POOL:
                self.MOCK_ETCD_POOL[key] = MockEtcd()
            self._mock_etcd = self.MOCK_ETCD_POOL[key]

    def __getattr__(self, attr):
        return getattr(self._mock_etcd, attr)

    def watch(self, key):
        return self._mock_etcd.watch(key, self)

    def close(self):
        self._mock_etcd.close(self)
