using System;
using System.Windows.Media;

namespace CellUniverse.Models.Managed {

    public sealed class MDataStorage : IDataStorage {

        int IDataStorage.Width {
            get {
                throw new NotImplementedException();
            }
        }

        int IDataStorage.Height {
            get {
                throw new NotImplementedException();
            }
        }

        int IDataStorage.LayersCount {
            get {
                throw new NotImplementedException();
            }
        }

        Color[,] IDataStorage.Result {
            get {
                throw new NotImplementedException();
            }
        }        
    }
}