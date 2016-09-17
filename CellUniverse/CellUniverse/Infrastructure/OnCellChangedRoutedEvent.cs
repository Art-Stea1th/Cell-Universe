using System;


namespace CellUniverse.Infrastructure {


    public delegate void OnCellChangedRoutedEvent(Tuple<int, int, bool> newState);
}