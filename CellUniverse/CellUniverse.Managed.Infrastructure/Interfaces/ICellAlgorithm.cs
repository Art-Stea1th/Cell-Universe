using System;
using System.Collections.Generic;

namespace CellUniverse.Managed.Infrastructure.Interfaces {

    public interface ICellAlgorithm {

        IEnumerable<Tuple<int, int, bool>> NextGeneration();
    }
}