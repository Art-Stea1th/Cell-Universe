using System;
using System.Collections.Generic;

namespace CellUniverse.Infrastructure.Interfaces {

    public interface ICellAlgorithm {

        IEnumerable<Tuple<short, short, bool>> NextGeneration();
        bool IsIdentical(bool[,] layer);
    }
}