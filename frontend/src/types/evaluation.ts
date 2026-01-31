/**
 * Kiki Agent Framework - 评估相关类型
 */

/**
 * 评估运行状态
 */
export type EvaluationStatus = 'pending' | 'running' | 'completed' | 'failed';

/**
 * 运行评估请求
 */
export interface RunEvaluationRequest {
  dataset_name: string;
  evaluators: string[];
  agent_type: string;
  session_id_prefix: string;
  max_entries?: number;
  categories?: string[];
  stream?: boolean;
}

/**
 * 运行评估响应
 */
export interface RunEvaluationResponse {
  run_id: string;
  status: EvaluationStatus;
  message: string;
}

/**
 * 评估结果条目
 */
export interface EvaluationResult {
  question: string;
  answer: string;
  expected?: string;
  score: number;
  feedback: string;
  passed: boolean;
}

/**
 * 评估摘要
 */
export interface EvaluationSummary {
  total: number;
  passed: number;
  failed: number;
  avg_score: number;
}

/**
 * 评估结果响应
 */
export interface EvaluationResultsResponse {
  run_id: string;
  status: EvaluationStatus;
  results?: EvaluationResult[];
  summary?: EvaluationSummary;
  error?: string;
  started_at?: string;
  completed_at?: string;
}

/**
 * 数据集信息
 */
export interface Dataset {
  name: string;
  description: string;
  entry_count: number;
  version: string;
  categories: string[];
}

/**
 * 数据集详情
 */
export interface DatasetDetail extends Dataset {
  entries?: DatasetEntry[];
}

/**
 * 数据集条目
 */
export interface DatasetEntry {
  id: string;
  question: string;
  expected_answer?: string;
  category?: string;
}

/**
 * 数据集列表查询参数
 */
export interface DatasetsQuery {
  category?: string;
}

/**
 * 评估结果列表查询参数
 */
export interface EvaluationResultsQuery {
  limit?: number;
}
