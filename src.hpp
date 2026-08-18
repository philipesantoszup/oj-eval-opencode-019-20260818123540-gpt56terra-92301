#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *query = rater.GetNextQuery();

    // Keep the supplied one-row matrices in HBM: later rounds reuse them.
    Matrix *key_matrix = keys[0];
    Matrix *value_matrix = values[0];
    for (size_t j = 1; j <= i; ++j) {
      Matrix *next_key = matrix_memory_allocator.Allocate("keys");
      Matrix *next_value = matrix_memory_allocator.Allocate("values");
      gpu_sim.Concat(key_matrix, keys[j], next_key, 0, kInGpuHbm);
      gpu_sim.Concat(value_matrix, values[j], next_value, 0, kInGpuHbm);
      if (j > 1) {
        gpu_sim.ReleaseMatrix(key_matrix);
        gpu_sim.ReleaseMatrix(value_matrix);
      }
      key_matrix = next_key;
      value_matrix = next_value;
    }

    // Round one needs disposable copies because its operands are inputs.
    if (i == 0) {
      Matrix *key_copy = matrix_memory_allocator.Allocate("key");
      Matrix *value_copy = matrix_memory_allocator.Allocate("value");
      gpu_sim.Copy(key_matrix, key_copy, kInGpuHbm);
      gpu_sim.Copy(value_matrix, value_copy, kInGpuHbm);
      key_matrix = key_copy;
      value_matrix = value_copy;
    }

    gpu_sim.MoveMatrixToSharedMem(query);
    gpu_sim.MoveMatrixToSharedMem(key_matrix);
    gpu_sim.MoveMatrixToSharedMem(value_matrix);

    gpu_sim.Transpose(key_matrix, kInSharedMemory);
    Matrix *scores = nullptr;
    constexpr size_t kFeatureBlock = 64;
    const size_t feature_count = query->GetColumnNum();
    for (size_t first_feature = 0; first_feature < feature_count;
         first_feature += kFeatureBlock) {
      const size_t last_feature =
          std::min(first_feature + kFeatureBlock, feature_count);
      Matrix *query_block = nullptr;
      Matrix *key_block = nullptr;
      for (size_t feature = first_feature; feature < last_feature; ++feature) {
        Matrix *query_column = matrix_memory_allocator.Allocate("query column");
        Matrix *key_row = matrix_memory_allocator.Allocate("key row");
        gpu_sim.GetColumn(query, feature, query_column, kInSharedMemory);
        gpu_sim.GetRow(key_matrix, feature, key_row, kInSharedMemory);
        if (query_block == nullptr) {
          query_block = query_column;
          key_block = key_row;
        } else {
          Matrix *next_query_block =
              matrix_memory_allocator.Allocate("query block");
          Matrix *next_key_block =
              matrix_memory_allocator.Allocate("key block");
          gpu_sim.Concat(query_block, query_column, next_query_block, 1,
                         kInSharedMemory);
          gpu_sim.Concat(key_block, key_row, next_key_block, 0,
                         kInSharedMemory);
          gpu_sim.ReleaseMatrix(query_block);
          gpu_sim.ReleaseMatrix(query_column);
          gpu_sim.ReleaseMatrix(key_block);
          gpu_sim.ReleaseMatrix(key_row);
          query_block = next_query_block;
          key_block = next_key_block;
        }
      }
      Matrix *partial_scores = matrix_memory_allocator.Allocate("partial scores");
      gpu_sim.MatMul(query_block, key_block, partial_scores);
      gpu_sim.ReleaseMatrix(query_block);
      gpu_sim.ReleaseMatrix(key_block);
      if (scores == nullptr) {
        scores = partial_scores;
      } else {
        Matrix *next_scores = matrix_memory_allocator.Allocate("scores");
        gpu_sim.MatAdd(scores, partial_scores, next_scores);
        gpu_sim.ReleaseMatrix(scores);
        gpu_sim.ReleaseMatrix(partial_scores);
        scores = next_scores;
      }
    }
    gpu_sim.ReleaseMatrix(query);
    gpu_sim.ReleaseMatrix(key_matrix);

    Matrix *probabilities = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      Matrix *score_row = matrix_memory_allocator.Allocate("score row");
      Matrix *exponentials = matrix_memory_allocator.Allocate("exponentials");
      Matrix *sum = matrix_memory_allocator.Allocate("sum");
      Matrix *probability_row =
          matrix_memory_allocator.Allocate("probability row");
      gpu_sim.GetRow(scores, row, score_row, kInSharedMemory);
      gpu_sim.MatExp(score_row, exponentials);
      gpu_sim.Sum(exponentials, sum);
      gpu_sim.MatDiv(exponentials, sum, probability_row);
      gpu_sim.ReleaseMatrix(score_row);
      gpu_sim.ReleaseMatrix(exponentials);
      gpu_sim.ReleaseMatrix(sum);

      if (probabilities == nullptr) {
        probabilities = probability_row;
      } else {
        Matrix *next_probabilities =
            matrix_memory_allocator.Allocate("probabilities");
        gpu_sim.Concat(probabilities, probability_row, next_probabilities, 0,
                       kInSharedMemory);
        gpu_sim.ReleaseMatrix(probabilities);
        gpu_sim.ReleaseMatrix(probability_row);
        probabilities = next_probabilities;
      }
    }
    gpu_sim.ReleaseMatrix(scores);

    Matrix *answer = matrix_memory_allocator.Allocate("answer");
    gpu_sim.MatMul(probabilities, value_matrix, answer);
    gpu_sim.ReleaseMatrix(probabilities);
    gpu_sim.ReleaseMatrix(value_matrix);
    gpu_sim.MoveMatrixToGpuHbm(answer);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
