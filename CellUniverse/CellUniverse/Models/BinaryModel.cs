using System;
using System.Collections.Generic;

namespace CellUniverse.Models {

    public class BinaryModel {

        private bool[,] _currentGeneration;

        public event OnCellChangedRoutedEvent OnCellChanged;

        public BinaryModel(int width, int height) {
            Initialize(width, height);
        }

        private void Initialize(int width, int height) {

            _currentGeneration = new bool[height, width];

            Random random = new Random();

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    _currentGeneration[y, x] = random.Next(2) == 1;
                }
            }
        }

        public void Next() {
            var nextGeneration = NextGeneration();
            foreach (var changedCell in GetDifference(_currentGeneration, nextGeneration)) {
                OnCellChanged?.Invoke(changedCell);
            }
            _currentGeneration = nextGeneration;
        }

        public void SimulateChangedAll() {
            for (int y = 0; y < _currentGeneration.GetLength(0); ++y) {
                for (int x = 0; x < _currentGeneration.GetLength(1); ++x) {
                    OnCellChanged?.Invoke(new Tuple<int, int, bool>(x, y, _currentGeneration[y, x]));
                }
            }
        }

        private IEnumerable<Tuple<int, int, bool>> GetDifference(bool[,] currentGeneration, bool[,] nextGeneration) {

            if (currentGeneration.GetLength(0) != nextGeneration.GetLength(0)
                || currentGeneration.GetLength(1) != nextGeneration.GetLength(1)) {
                throw new InvalidOperationException();
            }

            for (int y = 0; y < currentGeneration.GetLength(0); ++y) {
                for (int x = 0; x < currentGeneration.GetLength(1); ++x) {
                    if (currentGeneration[y, x] != nextGeneration[y, x]) {
                        yield return new Tuple<int, int, bool>(x, y, nextGeneration[y, x]);
                    }
                }
            }
        }

        private bool[,] NextGeneration() {
            return NextGenerationFromSource(_currentGeneration);
        }

        private bool[,] NextGenerationFromSource(bool[,] source) {

            var nextGeneration = new bool[source.GetLength(0), source.GetLength(1)];

            for (int y = 0; y < source.GetLength(0); y++) {
                for (int x = 0; x < source.GetLength(1); x++) {

                    int neighboursCount = CountNeighbours(source, x, y);

                    if ((neighboursCount == 2 || neighboursCount == 3) && IsAlive(source, x, y))
                        nextGeneration[y, x] = true;

                    if ((neighboursCount < 2 || neighboursCount > 3) && IsAlive(source, x, y))
                        nextGeneration[y, x] = false;

                    if (neighboursCount == 3 && !IsAlive(source, x, y))
                        nextGeneration[y, x] = true;
                }
            }
            return nextGeneration;
        }

        private int CountNeighbours(bool[,] generation, int posX, int posY) {

            int counter = 0;

            for (int y = posY - 1; y < posY + 2; y++) {
                for (int x = posX - 1; x < posX + 2; x++) {
                    if (x == posX && y == posY)
                        continue;

                    int py = y, px = x;

                    if (py == -1) py = generation.GetLength(0) - 1;
                    else if (py == generation.GetLength(0)) py = 0;

                    if (px == -1) px = generation.GetLength(1) - 1;
                    else if (px == generation.GetLength(1)) px = 0;                    

                    if (IsAlive(generation, px, py))
                        counter++;
                }
            }
            return counter;
        }

        private bool IsAlive(bool[,] generation, int posX, int posY) {
            return generation[posY, posX] ? true : false;
        }
    }
}