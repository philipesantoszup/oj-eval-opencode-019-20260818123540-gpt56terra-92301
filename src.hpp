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
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(query, key_matrix, scores);
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

    Matrix *answer = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      Matrix *probability_row =
          matrix_memory_allocator.Allocate("probability row");
      Matrix *answer_row = matrix_memory_allocator.Allocate("answer row");
      gpu_sim.GetRow(probabilities, row, probability_row, kInSharedMemory);
      gpu_sim.MatMul(probability_row, value_matrix, answer_row);
      gpu_sim.ReleaseMatrix(probability_row);

      if (answer == nullptr) {
        answer = answer_row;
      } else {
        Matrix *next_answer = matrix_memory_allocator.Allocate("answer");
        gpu_sim.Concat(answer, answer_row, next_answer, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer);
        gpu_sim.ReleaseMatrix(answer_row);
        answer = next_answer;
      }
    }
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
