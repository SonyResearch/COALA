import os
import unittest

from coala.tracking import client
from coala.tracking import metric
from coala.tracking import storage


class TrackingClientTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TrackingClientTest, self).__init__(*args, **kwargs)
        self._database = os.path.join(os.getcwd(), "tracker", "coala_test.db")
        self._store = storage.get_store(self._database)
        self._store.truncate_task_metric()
        self._store.truncate_round_metric()
        self._store.truncate_client_metric()

    def test_task(self):
        task_id = "test_task"
        conf = {"task_id": task_id}
        tracker = client.init_tracking(self._database)
        tracker.create_task(task_id, conf)
        m = tracker.get_task_metric(task_id)
        self.assertEqual(m.task_id, task_id)
        self.assertEqual(m.configuration, conf)
        m = tracker.get_task_metric("not_exist_task")
        self.assertEqual(m, None)

    def test_round(self):
        task_id = "test_round"
        round_id = 0
        want_metric = {"accuracy": 0.9, "loss": 0.1}
        want_train_upload_size = 10
        want_train_time = 20
        want_extra = {"extra": "information"}
        tracker = client.init_tracking(self._database)
        tracker.create_task(task_id)
        tracker.track_round(metric.TEST_METRIC, want_metric)
        m = tracker.get_round_metric(round_id, task_id)
        self.assertEqual(m.task_id, task_id)
        self.assertEqual(m.round_id, round_id)
        self.assertEqual(m.test_metric, want_metric)

        tracker.save_round(increment=False)
        round_metrics = self._store.get_round_metrics(task_id, [round_id])
        m = metric.RoundMetric.from_sql(next(round_metrics))
        self.assertEqual(m.task_id, task_id)
        self.assertEqual(m.round_id, round_id)
        self.assertEqual(m.test_metric, want_metric)

        # round 1
        tracker.set_round(round_id + 1)
        tracker.track_round(metric.TRAIN_UPLOAD_SIZE, want_train_upload_size)
        tracker.track_round(metric.EXTRA, want_extra)
        tracker.save_round()
        round_metrics = self._store.get_round_metrics(task_id, [round_id + 1])
        m = metric.RoundMetric.from_sql(next(round_metrics))
        self.assertEqual(m.task_id, task_id)
        self.assertEqual(m.round_id, round_id + 1)
        self.assertEqual(m.test_accuracy, 0)
        self.assertEqual(m.train_upload_size, want_train_upload_size)
        self.assertEqual(m.extra["extra"], want_extra["extra"])

        # round 2
        tracker.track_round(metric.TRAIN_TIME, want_train_time)
        m = tracker.get_round_metric(round_id + 2, task_id)
        self.assertEqual(m.task_id, task_id)
        self.assertEqual(m.round_id, round_id + 2)
        self.assertEqual(m.train_time, want_train_time)
        self.assertEqual(m.test_accuracy, 0)
        self.assertEqual(m.extra, {})

    def test_client(self):
        task_id = "test_client"
        round_id = 1
        client_id = "client_id_test"
        client_id_2 = "client_id_test_2"
        want_metric = {"accuracy": 0.9123456789, "mAP": 0.8, "loss": 0.1}
        want_rank1 = 0.7
        want_extra = {"extra": "information"}

        # test error
        tracker = client.init_tracking(self._database)
        self.assertRaises(LookupError, tracker.create_client, client_id)
        self.assertRaises(LookupError, tracker.track_client, metric.TEST_METRIC, [want_metric])

        # test track and get client
        tracker.set_client_context(task_id, round_id, client_id)
        tracker.track_client(metric.TRAIN_METRIC, [want_metric])
        m = tracker.get_client_metric(client_id, round_id, task_id)
        self.assertEqual(m.train_metric, [want_metric])

        # test save client
        tracker.save_client()
        client_metrics = self._store.get_client_metrics(task_id, round_id, [client_id])
        m = metric.ClientMetric.from_sql(next(client_metrics))
        self.assertEqual(m.task_id, task_id)
        self.assertEqual(m.round_id, round_id)
        self.assertEqual(m.client_id, client_id)
        self.assertEqual(m.train_metric, [want_metric])
        self._store.truncate_client_metric()

        # test save multiple clients
        tracker2 = client.init_tracking(self._database)
        tracker2.set_client_context(task_id, round_id, client_id_2)
        tracker2.track_client("rank1", want_rank1)
        tracker2.track_client(metric.EXTRA, want_extra)

        client_metrics = [tracker.get_client_metric(), tracker2.get_client_metric()]
        tracker.save_clients(client_metrics)
        results = self._store.get_client_metrics(task_id, round_id, [client_id, client_id_2])
        metrics = [metric.ClientMetric.from_sql(r) for r in results]
        self.assertEqual(len(metrics), 2)
        self.assertEqual(len(metrics[1].extra), 2)
        self.assertEqual(metrics[1].task_id, task_id)
        self.assertEqual(metrics[1].round_id, round_id)
        self.assertEqual(metrics[1].client_id, client_id_2)
        self.assertEqual(metrics[1].extra["rank1"], want_rank1)


if __name__ == '__main__':
    unittest.main()
