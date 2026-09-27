import asyncio
import io

import aiohttp
import picologging

from benchmarks import slo_split_matrix


def test_failed_connection_keeps_error_record_and_picologging_traceback():
    stream = io.StringIO()
    handler = picologging.StreamHandler(stream)
    logger = slo_split_matrix.logger
    assert isinstance(logger, picologging.Logger)
    logger.addHandler(handler)
    try:

        async def exercise():
            async with aiohttp.ClientSession() as session:
                return await slo_split_matrix.one_request(
                    session, "http://127.0.0.1:0", "unused", "test", 1, 2
                )

        result = asyncio.run(exercise())
    finally:
        logger.removeHandler(handler)
        handler.close()
    assert not result["ok"]
    assert result["error"]
    assert "Streaming measurement failed" in stream.getvalue()
    assert "Traceback" in stream.getvalue()
