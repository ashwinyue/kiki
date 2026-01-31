/**
 * Kiki Agent Framework - 评估服务
 */

import api, { request } from './api';
import type {
  RunEvaluationRequest,
  RunEvaluationResponse,
  EvaluationResultsResponse,
  Dataset,
  DatasetDetail,
  DatasetsQuery,
  EvaluationResultsQuery,
} from '@/types/evaluation';

/**
 * 运行评估
 */
export async function runEvaluation(data: RunEvaluationRequest): Promise<RunEvaluationResponse> {
  return request(() =>
    api.post<RunEvaluationResponse>('/evaluation/run', data)
  );
}

/**
 * 获取评估结果
 */
export async function getEvaluationResults(runId: string): Promise<EvaluationResultsResponse> {
  return request(() =>
    api.get<EvaluationResultsResponse>(`/evaluation/results/${runId}`)
  );
}

/**
 * 获取评估结果 Markdown
 */
export async function getEvaluationResultsMarkdown(runId: string): Promise<string> {
  return request(() =>
    api.get<string>(`/evaluation/results/${runId}/markdown`)
  );
}

/**
 * 获取评估结果列表
 */
export async function listEvaluationResults(
  query?: EvaluationResultsQuery
): Promise<EvaluationResultsResponse[]> {
  return request(() =>
    api.get<EvaluationResultsResponse[]>('/evaluation/results', { params: query })
  );
}

/**
 * 删除评估结果
 */
export async function deleteEvaluationResults(runId: string): Promise<void> {
  return request(() =>
    api.delete(`/evaluation/results/${runId}`)
  );
}

/**
 * 获取数据集列表
 */
export async function getDatasets(query?: DatasetsQuery): Promise<Dataset[]> {
  return request(() =>
    api.get<Dataset[]>('/evaluation/datasets', { params: query })
  );
}

/**
 * 获取数据集详情
 */
export async function getDataset(datasetName: string): Promise<DatasetDetail> {
  return request(() =>
    api.get<DatasetDetail>(`/evaluation/datasets/${datasetName}`)
  );
}

/**
 * 获取数据集条目
 */
export async function getDatasetEntries(
  datasetName: string,
  query?: DatasetsQuery
): Promise<DatasetDetail['entries']> {
  return request(() =>
    api.get<DatasetDetail['entries']>(`/evaluation/datasets/${datasetName}/entries`, {
      params: query,
    })
  );
}

export default {
  runEvaluation,
  getEvaluationResults,
  getEvaluationResultsMarkdown,
  listEvaluationResults,
  deleteEvaluationResults,
  getDatasets,
  getDataset,
  getDatasetEntries,
};
