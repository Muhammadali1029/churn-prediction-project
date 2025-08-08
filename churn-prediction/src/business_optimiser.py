
"""
Business driven model optimisation.
Real ML engineering is about business impact, not just metrics.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .logger_config import setup_logger

logger = setup_logger(__name__)

class ChurnBusinessOptimiser:
    """Optimise models for business metrics, not just ML metrics"""
    
    def __init__(self, monthly_revenue_per_customer=70, retention_cost=10, 
                 retention_success_rate=0.3):
        """
        Args:
            monthly_revenue_per_customer: Average monthly revenue
            retention_cost: Cost to run retention campaign per customer
            retention_success_rate: Probability of retaining a churning customer
        """
        self.monthly_revenue = monthly_revenue_per_customer
        self.retention_cost = retention_cost
        self.retention_success_rate = retention_success_rate
        
    def calculate_business_value(self, y_true, y_pred_proba, threshold=0.5):
        """Calculate expected business value of the model"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Business logic:
        # - True Positive: We correctly identify churner, attempt retention
        #   Value = (retention_success_rate * monthly_revenue * 12) - retention_cost
        # - False Positive: We incorrectly target loyal customer
        #   Cost = retention_cost (wasted campaign)
        # - False Negative: We miss a churner
        #   Cost = monthly_revenue * 12 (lost annual revenue)
        # - True Negative: Correctly identify loyal customer
        #   Value = 0 (no action needed)
        
        value_per_tp = tp * (self.retention_success_rate * self.monthly_revenue * 12) - self.retention_cost

         # If retention isn't profitable, we shouldn't target anyone
        if value_per_tp < 0:
            logger.warning(f"Retention not profitable! Value per TP: ${value_per_tp:.2f}")
    

        value_tp = tp * value_per_tp
        cost_fp = fp * self.retention_cost
        cost_fn = fn * self.monthly_revenue * 12
        
        total_value = value_tp - cost_fp - cost_fn
        
        # Per customer value
        value_per_customer = total_value / len(y_true)
        
        metrics = {
            'threshold': threshold,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'total_business_value': total_value,
            'value_per_customer': value_per_customer,
            'customers_to_target': tp + fp,
            'value_per_tp': value_per_tp,
            'annual_revenue': self.monthly_revenue * 12,
            'retention_value': self.retention_success_rate * self.monthly_revenue * 12
        }
        
        return metrics
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Find the threshold that maximizes business value"""
        # Use finer granularity for threshold search
        thresholds = np.arange(0.05, 0.95, 0.005)
        results = []
        
        for threshold in thresholds:
            metrics = self.calculate_business_value(y_true, y_pred_proba, threshold)
            results.append(metrics)
        
        df_results = pd.DataFrame(results)
        
        # Debug: print value at different thresholds
        sample_thresholds = [0.1, 0.3, 0.5, 0.7]
        print("\nBusiness value at different thresholds:")
        for t in sample_thresholds:
            idx = df_results[df_results['threshold'].round(2) == t].index[0]
            print(f"Threshold {t}: Value=${df_results.loc[idx, 'total_business_value']:,.2f}, "
                f"Target={df_results.loc[idx, 'customers_to_target']}, "
                f"TP={df_results.loc[idx, 'true_positives']}")
        
        optimal_idx = df_results['total_business_value'].idxmax()
        optimal_threshold = df_results.loc[optimal_idx, 'threshold']
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f}", extra={
            'max_value': df_results.loc[optimal_idx, 'total_business_value'],
            'customers_to_target': df_results.loc[optimal_idx, 'customers_to_target'],
            'precision': df_results.loc[optimal_idx, 'precision'],
            'recall': df_results.loc[optimal_idx, 'recall']
        })
    
        return optimal_threshold, df_results
    
    def plot_threshold_analysis(self, df_results):
        """Visualize how threshold affects business metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Business value vs threshold
        axes[0, 0].plot(df_results['threshold'], df_results['total_business_value'])
        axes[0, 0].axvline(df_results.loc[df_results['total_business_value'].idxmax(), 'threshold'], 
                          color='r', linestyle='--', label='Optimal')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Total Business Value ($)')
        axes[0, 0].set_title('Business Value vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Precision-Recall trade-off
        axes[0, 1].plot(df_results['threshold'], df_results['precision'], label='Precision')
        axes[0, 1].plot(df_results['threshold'], df_results['recall'], label='Recall')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Precision-Recall Trade-off')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Customers to target
        axes[1, 0].plot(df_results['threshold'], df_results['customers_to_target'])
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].set_title('Customers to Target for Retention')
        axes[1, 0].grid(True)
        
        # Value per customer
        axes[1, 1].plot(df_results['threshold'], df_results['value_per_customer'])
        axes[1, 1].axhline(0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Value per Customer ($)')
        axes[1, 1].set_title('Expected Value per Customer')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def create_business_report(self, model, X_test, y_test, model_name="Model"):
        """Create a business-focused report"""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        optimal_threshold, df_results = self.find_optimal_threshold(y_test, y_pred_proba)
        
        # Get metrics at optimal threshold
        optimal_metrics = self.calculate_business_value(y_test, y_pred_proba, optimal_threshold)
        
        # Create report
        report = f"""
        BUSINESS IMPACT REPORT - {model_name}
        =====================================
        
        Model Performance (Optimal Threshold: {optimal_threshold:.3f}):
        - Customers to target: {optimal_metrics['customers_to_target']:,}
        - Precision: {optimal_metrics['precision']:.1%} (of targeted customers, this % will actually churn)
        - Recall: {optimal_metrics['recall']:.1%} (of all churners, we'll catch this %)
        
        Financial Impact (Annual):
        - Total expected value: ${optimal_metrics['total_business_value']:,.2f}
        - Value per customer: ${optimal_metrics['value_per_customer']:.2f}
        - ROI: {(optimal_metrics['total_business_value'] / (optimal_metrics['customers_to_target'] * self.retention_cost)):.1%}
        
        Recommendations:
        1. Target customers with churn probability > {optimal_threshold:.1%}
        2. This will identify {optimal_metrics['true_positives']} actual churners
        3. Expected to retain {int(optimal_metrics['true_positives'] * self.retention_success_rate)} customers
        4. Budget needed: ${optimal_metrics['customers_to_target'] * self.retention_cost:,.2f}
        """
        
        print(report)
        
        # Visualize
        self.plot_threshold_analysis(df_results)
        
        return optimal_metrics, df_results