namespace CellUniverse.Models.TheGameOfLife {


    internal sealed class Cell {

        private bool isAlive;
        private bool isAliveInTheNextGeneration;

        // Critical to the speed. Using collections in 2x-3x times slower.
        private Cell lt, lc, lb, ct, cb, rt, rc, rb;

        internal int X { get; private set; }
        internal int Y { get; private set; }

        public static implicit operator bool(Cell cell) {
            return cell.isAlive;
        }

        public static implicit operator Cell(bool cell) {
            return new Cell(cell);
        }

        private Cell(bool state = false) {
            X = 0; Y = 0;
            isAlive = state;
            isAliveInTheNextGeneration = false;            
        }

        internal bool CalculateNextState() {

            byte aliveNeighboursCount = 0;

            if (lt) { ++aliveNeighboursCount; }
            if (lc) { ++aliveNeighboursCount; }
            if (lb) { ++aliveNeighboursCount; }

            if (ct) { ++aliveNeighboursCount; }
            if (cb) { ++aliveNeighboursCount; }

            if (rt) { ++aliveNeighboursCount; }
            if (rc) { ++aliveNeighboursCount; }
            if (rb) { ++aliveNeighboursCount; }

            if ((aliveNeighboursCount == 2 || aliveNeighboursCount == 3) && isAlive) { isAliveInTheNextGeneration = true; }
            if (aliveNeighboursCount == 3 && !isAlive) { isAliveInTheNextGeneration = true; }

            return isAliveInTheNextGeneration;
        }

        internal void ApplyNextState() {
            isAlive = isAliveInTheNextGeneration;
            isAliveInTheNextGeneration = false;
        }

        internal void ConnectToNeighbours(int x, int y, Cell[] container, int virtualWidth, int virtualHeight) {

            X = x; Y = y;

            lt = GetNeighbourAtIndex(container, X - 1, Y - 1, virtualWidth, virtualHeight);
            lc = GetNeighbourAtIndex(container, X - 1, Y, virtualWidth, virtualHeight);
            lb = GetNeighbourAtIndex(container, X - 1, Y + 1, virtualWidth, virtualHeight);

            ct = GetNeighbourAtIndex(container, X, Y - 1, virtualWidth, virtualHeight);
            cb = GetNeighbourAtIndex(container, X, Y + 1, virtualWidth, virtualHeight);

            rt = GetNeighbourAtIndex(container, X + 1, Y - 1, virtualWidth, virtualHeight);
            rc = GetNeighbourAtIndex(container, X + 1, Y, virtualWidth, virtualHeight);
            rb = GetNeighbourAtIndex(container, X + 1, Y + 1, virtualWidth, virtualHeight);
        }

        private Cell GetNeighbourAtIndex(Cell[] container, int x, int y, int width, int height) {

            int fixedX = FixPoint(x, width);
            int fixedY = FixPoint(y, height);
            int realIndex = fixedX + fixedY * width;

            return container[realIndex];
        }

        private int FixPoint(int point, int vectorLength) {

            if (point < 0) { return vectorLength + point; }
            if (point >= vectorLength) { return point - vectorLength; }

            return point;
        }
    }
}