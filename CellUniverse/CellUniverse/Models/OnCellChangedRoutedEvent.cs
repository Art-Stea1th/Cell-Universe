using System;


namespace CellUniverse.Models {


    public delegate void OnCellChangedRoutedEvent(Tuple<int, int, bool> newState);
}