﻿using System;
using System.ComponentModel;
using System.Linq.Expressions;

namespace RCVL.IntegrationTests {

    public abstract class ViewModelBase : INotifyPropertyChanged, IDisposable {

        protected string GetMemberName<T, TValue>(Expression<Func<T, TValue>> memberAccess) {
            return ((MemberExpression)memberAccess.Body).Member.Name;
        }

        #region impl. INotifyPropertyChanged

        protected ViewModelBase() { }

        public event PropertyChangedEventHandler PropertyChanged;

        public virtual void OnPropertyChanged(string propertyName) {
            PropertyChangedEventHandler handler = this.PropertyChanged;
            if (handler != null) {
                handler.Invoke(this, new PropertyChangedEventArgs(propertyName));
            }
        }

        #endregion

        #region impl. IDisposable

        public void Dispose() {
            this.OnDispose();
        }

        protected virtual void OnDispose() { }

        #endregion
    }
}