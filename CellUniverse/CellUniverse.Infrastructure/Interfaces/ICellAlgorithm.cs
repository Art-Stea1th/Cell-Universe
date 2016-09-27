using System;
using System.Collections.Generic;

namespace CellUniverse.Infrastructure.Interfaces {

    public interface ICellAlgorithm {

        IEnumerable<Tuple<int, int, bool>> NextGeneration();
    }
}