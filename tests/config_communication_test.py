from unittest.mock import Mock

from dlio_benchmark.utils.config import ConfigArguments, LoadConfig


def _config_arguments(world_size):
    args = object.__new__(ConfigArguments)
    args.comm_size = world_size
    args.communication = False
    args.logger = Mock()
    return args


def test_load_config_disables_communication_for_one_rank():
    args = _config_arguments(world_size=1)

    LoadConfig(args, {"train": {"communication": True}})

    assert args.communication is False
    args.logger.warning.assert_called_once()
    assert "MPI world size is 1" in args.logger.warning.call_args.args[0]


def test_load_config_preserves_communication_for_multiple_ranks():
    args = _config_arguments(world_size=2)

    LoadConfig(args, {"train": {"communication": True}})

    assert args.communication is True
    args.logger.warning.assert_not_called()


def test_load_config_keeps_disabled_communication_off():
    args = _config_arguments(world_size=2)

    LoadConfig(args, {"train": {"communication": False}})

    assert args.communication is False
    args.logger.warning.assert_not_called()
