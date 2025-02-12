Get Job Summaries
=================

.. autofunction:: netspresso.np_qai.base.NPQAIBase.get_job_summaries


Example
-------

.. code-block:: python

    from netspresso import NPQAI

    QAI_HUB_API_TOKEN = "YOUR_QAI_HUB_API_TOKEN"
    np_qai = NPQAI(api_token=QAI_HUB_API_TOKEN)
    job_summaries = np_qai.get_job_summaries()
