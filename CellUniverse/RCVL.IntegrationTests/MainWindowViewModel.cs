using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RCVL.IntegrationTests {

    public class MainWindowViewModel {

        public List<bool[,]> LayeredData { get; set; }

        public MainWindowViewModel() {
            LayeredData = new List<bool[,]>();
            LayeredData.Add(new bool[1, 1]);
            LayeredData[0][0, 0] = true;
        }
    }
}